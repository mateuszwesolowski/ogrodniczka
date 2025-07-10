import streamlit as st
import pandas as pd
import calendar
from datetime import datetime, timedelta
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from openai import OpenAI
from qdrant_client.models import PayloadSchemaType

# Konfiguracja strony
st.set_page_config(
    page_title="Kalendarz Ogrodniczki Pauli",
    page_icon="🌱",
    layout="wide"
)

# Inicjalizacja klienta OpenAI
@st.cache_resource
def init_openai():
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# Funkcja agenta OpenAI do generowania kalendarza upraw
def wygeneruj_kalendarz_upraw(client_openai, nazwa_uprawy, rok=2025):
    if not client_openai:
        return None
    
    prompt = f"""
    Jesteś ekspertem ogrodnikiem. Przygotuj kompletny kalendarz upraw dla rośliny: {nazwa_uprawy} na rok {rok}.
    
    Zwróć odpowiedź w formacie JSON z następującą strukturą:
    {{
        "nazwa": "Nazwa rośliny",
        "zadania": [
            {{
                "data": "YYYY-MM-DD",
                "opis": "Opis zadania"
            }}
        ]
    }}
    
    Uwzględnij wszystkie kluczowe etapy uprawy:
    - Przygotowanie ziemi/podłoża
    - Wysiew nasion lub sadzenie rozsady
    - Pielęgnację (podlewanie, nawożenie, odchwaszczanie)
    - Formowanie (przycinanie, podpieranie)
    - Zbiór i przechowywanie
    - Przygotowanie do zimy (jeśli dotyczy)
    
    Uwzględnij klimat umiarkowany (Polska) i podaj realistyczne daty dla każdego zadania.
    Każde zadanie powinno mieć konkretny, praktyczny opis.
    
    Zwróć tylko JSON, bez dodatkowych komentarzy.
    """
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem ogrodnikiem i zwracasz odpowiedzi wyłącznie w formacie JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        # Parsuj odpowiedź JSON
        response_text = response.choices[0].message.content.strip()
        
        # Usuń potencjalne markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        kalendarz = json.loads(response_text)
        return kalendarz
        
    except Exception as e:
        st.error(f"Błąd generowania kalendarza: {e}")
        return None
    
# Inicjalizacja klienta Qdrant
@st.cache_resource
def init_qdrant():
    try:
        url = st.secrets["QDRANT_URL"]
        api_key = st.secrets.get("QDRANT_API_KEY", None)

        client = QdrantClient(url=url, api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Nie można połączyć z Qdrant: {e}")
        st.stop()

# Inicjalizacja kolekcji
def init_collections(client):
    collection_name = "kalendarz_ogrodnika"
    
    try:
        # Sprawdź czy kolekcja istnieje
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if not collection_exists:
            # Utwórz kolekcję
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                optimizers_config=None,
                on_disk_payload=True
            )
            st.success("Utworzono nową kolekcję")
            # Tworzymy indeksy tylko dla nowej kolekcji i pokazujemy komunikaty
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                st.success("Utworzono indeks dla pola 'type'")
            except Exception as e:
                if "already exists" in str(e).lower():
                    pass
                else:
                    st.warning(f"Indeks 'type': {e}")
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="uprawa_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                st.success("Utworzono indeks dla pola 'uprawa_id'")
            except Exception as e:
                if "already exists" in str(e).lower():
                    pass
                else:
                    st.warning(f"Indeks 'uprawa_id': {e}")
        else:
            # Jeśli kolekcja już istnieje, próbujemy utworzyć indeksy, ale nie pokazujemy komunikatów
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    pass
                else:
                    st.warning(f"Indeks 'type': {e}")
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="uprawa_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    pass
                else:
                    st.warning(f"Indeks 'uprawa_id': {e}")

        # Dodaj domyślne uprawy tylko dla nowej kolekcji
        if not collection_exists:
            default_uprawy = {
                'pomidory': {
                    'nazwa': 'Pomidory',
                    'zadania': [
                        {'data': '2025-03-15', 'opis': 'Wysiew nasion na rozsadę'},
                        {'data': '2025-05-15', 'opis': 'Przesadzanie rozsady do gruntu'},
                        {'data': '2025-06-01', 'opis': 'Podlewanie i nawożenie'},
                        {'data': '2025-07-01', 'opis': 'Zbieranie pierwszych owoców'},
                        {'data': '2025-08-15', 'opis': 'Regularne zbieranie owoców'}
                    ]
                },
                'marchew': {
                    'nazwa': 'Marchew',
                    'zadania': [
                        {'data': '2025-04-01', 'opis': 'Wysiew nasion do gruntu'},
                        {'data': '2025-05-01', 'opis': 'Przerzedzanie siewek'},
                        {'data': '2025-06-15', 'opis': 'Regularne podlewanie'},
                        {'data': '2025-09-01', 'opis': 'Zbieranie marchewki'}
                    ]
                },
                'ogorki': {
                    'nazwa': 'Ogórki',
                    'zadania': [
                        {'data': '2025-04-15', 'opis': 'Wysiew nasion na rozsadę'},
                        {'data': '2025-05-20', 'opis': 'Przesadzanie do gruntu'},
                        {'data': '2025-06-10', 'opis': 'Podpieranie roślin'},
                        {'data': '2025-07-15', 'opis': 'Zbieranie owoców'}
                    ]
                }
            }
            
            for uprawa_id, uprawa_data in default_uprawy.items():
                dodaj_uprawe_do_bazy(client, uprawa_id, uprawa_data)
            
            st.success("Dodano domyślne uprawy")
        
        return collection_name
    except Exception as e:
        st.error(f"Błąd inicjalizacji kolekcji: {e}")
        st.stop()

# Funkcje do operacji na bazie danych

def dodaj_uprawe_do_bazy(client, uprawa_id, uprawa_data):
    collection_name = "kalendarz_ogrodnika"
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=[1.0],  # Dummy vector
        payload={
            "type": "uprawa",
            "uprawa_id": uprawa_id,
            "nazwa": uprawa_data['nazwa'],
            "zadania": uprawa_data['zadania']
        }
    )
    client.upsert(collection_name=collection_name, points=[point])

def pobierz_uprawy_z_bazy(client):
    collection_name = "kalendarz_ogrodnika"
    try:
        # Pobierz wszystkie uprawy
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "type", "match": {"value": "uprawa"}}]},
            limit=1000
        )
        uprawy = {}
        for point in results[0]:
            payload = point.payload
            uprawy[payload['uprawa_id']] = {
                'nazwa': payload['nazwa'],
                'zadania': payload['zadania']
            }
        return uprawy
    except Exception as e:
        st.error(f"Błąd pobierania upraw z bazy: {e}")
        return {}

def pobierz_wybrane_uprawy_z_bazy(client):
    collection_name = "kalendarz_ogrodnika"
    try:
        # Pobierz ustawienia wybranych upraw
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "type", "match": {"value": "ustawienia"}}]},
            limit=1
        )
        if results[0]:
            return results[0][0].payload.get('wybrane_uprawy', [])
        else:
            return []
    except Exception as e:
        return []

def zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy):
    collection_name = "kalendarz_ogrodnika"
    try:
        # Usuń stare ustawienia
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                    key="type",
                    match=MatchValue(value="ustawienia")
                    )
                ]       
            )
        )
        # Dodaj nowe ustawienia
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[1.0],
            payload={
                "type": "ustawienia",
                "wybrane_uprawy": wybrane_uprawy
            }
        )
        client.upsert(collection_name=collection_name, points=[point])
    except Exception as e:
        st.error(f"Błąd zapisywania ustawień: {e}")

def usun_uprawe_z_bazy(client, uprawa_id):
    collection_name = "kalendarz_ogrodnika"
    
    try:
        client.delete(
            collection_name=collection_name,
            points_selector={"filter": {"must": [
                {"key": "type", "match": {"value": "uprawa"}},
                {"key": "uprawa_id", "match": {"value": uprawa_id}}
            ]}}
        )
    except Exception as e:
        st.error(f"Błąd usuwania uprawy: {e}")

# Inicjalizacja
client = init_qdrant()
client_openai = init_openai()
collection_name = init_collections(client)

# --- Pobieranie upraw zawsze na bieżąco (na początku pętli) ---
uprawy = pobierz_uprawy_z_bazy(client)
wybrane_uprawy = pobierz_wybrane_uprawy_z_bazy(client)
if uprawy and not wybrane_uprawy:
    wybrane_uprawy = list(uprawy.keys())
    zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy)

# Funkcja do pobierania zadań na dany dzień
def pobierz_zadania_na_dzien(data, uprawy, wybrane_uprawy):
    zadania = []
    for uprawa_id, uprawa in uprawy.items():
        if uprawa_id in wybrane_uprawy:
            for zadanie in uprawa['zadania']:
                if zadanie['data'] == data.strftime('%Y-%m-%d'):
                    zadania.append({
                        'uprawa': uprawa['nazwa'],
                        'opis': zadanie['opis']
                    })
    return zadania

# Funkcja do pobierania zadań w zakresie dat
def pobierz_zadania_w_zakresie(data_od, data_do, uprawy, wybrane_uprawy):
    zadania = []
    for uprawa_id, uprawa in uprawy.items():
        if uprawa_id in wybrane_uprawy:
            for zadanie in uprawa['zadania']:
                zadanie_data = datetime.strptime(zadanie['data'], '%Y-%m-%d').date()
                if data_od <= zadanie_data <= data_do:
                    zadania.append({
                        'data': zadanie_data,
                        'uprawa': uprawa['nazwa'],
                        'opis': zadanie['opis']
                    })
    return sorted(zadania, key=lambda x: x['data'])

# --- KALENDARZ: emoji kwadraty ---
def rysuj_kalendarz(rok, miesiac, uprawy, wybrane_uprawy):
    cal = calendar.monthcalendar(rok, miesiac)
    nazwa_miesiaca = calendar.month_name[miesiac]
    st.subheader(f"{nazwa_miesiaca} {rok}")

    dni_tygodnia = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Nie']
    cols = st.columns(7)
    for i, dzien in enumerate(dni_tygodnia):
        cols[i].markdown(f"**{dzien}**")

    for tydzien in cal:
        cols = st.columns(7)
        for i, dzien in enumerate(tydzien):
            if dzien == 0:
                cols[i].markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
            else:
                data = datetime(rok, miesiac, dzien).date()
                # Zbierz emoji upraw z zadaniami na ten dzień
                emoji_list = []
                for uprawa_id, uprawa in uprawy.items():
                    if uprawa_id in wybrane_uprawy:
                        for zad in uprawa['zadania']:
                            if zad['data'] == data.strftime('%Y-%m-%d'):
                                emoji_list.append(uprawa.get('emoji', '🟩'))
                                break
                key_btn = f"day_{rok}_{miesiac}_{dzien}"
                is_selected = st.session_state.get('selected_day') == str(data)
                # Emoji dla każdej uprawy z zadaniem
                label = f"{''.join(emoji_list)} {dzien}" if emoji_list else f"{dzien}"
                if cols[i].button(label, key=key_btn, use_container_width=True):
                    st.session_state['context_day'] = str(data)
                    st.session_state['context_action'] = 'menu'
                    st.session_state['selected_day'] = str(data)
    # Legenda pod kalendarzem
    legenda = []
    for uprawa_id in wybrane_uprawy:
        uprawa = uprawy.get(uprawa_id)
        if uprawa:
            emoji = uprawa.get('emoji', '🟩')
            nazwa = uprawa['nazwa']
            legenda.append(f"{emoji} {nazwa}")
    if legenda:
        st.markdown("**Legenda:** " + " &nbsp; ".join(legenda))


# Główny interfejs
st.title("🌱 Kalendarz Ogrodniczki Pauli")

# Sidebar - zarządzanie uprawami
with st.sidebar:
    st.markdown("### Nawigacja")
    col_kal, col_upr = st.columns(2)
    if col_kal.button("Kalendarz", key="btn_kalendarz"):
        st.session_state['main_view'] = 'kalendarz'
    if col_upr.button("Zarządzaj uprawami", key="btn_uprawy"):
        st.session_state['main_view'] = 'uprawy'
    st.divider()
    
    st.header("Zarządzanie uprawami")
    
    # Status połączenia z bazą
    st.success("🔗 Połączono z bazą Qdrant")
    
    # Wybór upraw do wyświetlenia
    st.subheader("Wybierz uprawy")
    wszystkie_uprawy = list(uprawy.keys())
    nazwy_upraw = [uprawy[u]['nazwa'] for u in wszystkie_uprawy]
    
    wybrane_nazwy = st.multiselect(
        "Uprawy do wyświetlenia:",
        nazwy_upraw,
        default=[uprawy[u]['nazwa'] for u in wybrane_uprawy if u in uprawy]
    )
    
    # Aktualizacja wybranych upraw
    nowe_wybrane_uprawy = [
        u for u in wszystkie_uprawy 
        if uprawy[u]['nazwa'] in wybrane_nazwy
    ]
    
    if nowe_wybrane_uprawy != wybrane_uprawy:
        wybrane_uprawy = nowe_wybrane_uprawy
        zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy)
        st.rerun()
    
    st.divider()
    
    # Mój Pomocnik - Agent OpenAI
    st.subheader("🤖 Mój Pomocnik")
    
    if client_openai:
        st.success("✅ Agent OpenAI gotowy do pracy")
        
        with st.form("pomocnik_upraw"):
            st.write("**Powiedz mi jaką uprawę chcesz dodać, a ja stworzę kompletny kalendarz!**")
            
            nazwa_uprawy_ai = st.text_input(
                "Nazwa rośliny/uprawy:",
                placeholder="np. bazylia, truskawki, róże, lawenda..."
            )
            
            rok_uprawy = st.selectbox(
                "Rok uprawy:",
                options=[2025, 2026, 2027],
                index=0
            )
            
            col1, col2 = st.columns(2)
            with col1:
                generuj_btn = st.form_submit_button("🌱 Wygeneruj kalendarz", type="primary")
            with col2:
                if st.form_submit_button("💡 Pokaż przykłady"):
                    st.info("""
                    **Przykłady upraw:**
                    - Warzywa: pomidory, ogórki, marchew, ziemniaki
                    - Zioła: bazylia, tymianek, rozmaryn, mięta
                    - Kwiaty: róże, tulipany, słoneczniki
                    - Owoce: truskawki, maliny, jabłonie
                    """)
            
            if generuj_btn and nazwa_uprawy_ai:
                with st.spinner(f"🤖 Generuję kalendarz upraw dla: {nazwa_uprawy_ai}..."):
                    kalendarz_ai = wygeneruj_kalendarz_upraw(client_openai, nazwa_uprawy_ai, rok_uprawy)
                    
                    if kalendarz_ai and 'zadania' in kalendarz_ai:
                        # Stwórz uprawa_id
                        uprawa_id = nazwa_uprawy_ai.lower().replace(' ', '_').replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
                        
                        # Dodaj do bazy
                        dodaj_uprawe_do_bazy(client, uprawa_id, {
                            'nazwa': kalendarz_ai.get('nazwa', nazwa_uprawy_ai),
                            'zadania': kalendarz_ai['zadania']
                        })
                        
                        # Dodaj do wybranych upraw
                        if uprawa_id not in wybrane_uprawy:
                            wybrane_uprawy.append(uprawa_id)
                            zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy)
                        
                        st.success(f"✅ Kalendarz dla '{kalendarz_ai.get('nazwa', nazwa_uprawy_ai)}' został wygenerowany i dodany!")
                        st.info(f"📅 Dodano {len(kalendarz_ai['zadania'])} zadań ogrodniczych")
                        
                        # Pokaż podgląd zadań
                        with st.expander("👀 Podgląd wygenerowanych zadań"):
                            for i, zadanie in enumerate(kalendarz_ai['zadania'][:5]):  # Pokaż pierwsze 5 zadań
                                st.write(f"**{zadanie['data']}**: {zadanie['opis']}")
                            if len(kalendarz_ai['zadania']) > 5:
                                st.write(f"... i {len(kalendarz_ai['zadania']) - 5} innych zadań")
                        
                        st.rerun()
                    else:
                        st.error("❌ Nie udało się wygenerować kalendarza. Spróbuj ponownie z inną nazwą rośliny.")
            
            elif generuj_btn and not nazwa_uprawy_ai:
                st.error("⚠️ Podaj nazwę rośliny/uprawy!")
    else:
        st.warning("⚠️ Brak klucza API OpenAI")
        st.info("""
        Aby używać funkcji 'Mój Pomocnik':
        1. Dodaj klucz API OpenAI do pliku `.streamlit/secrets.toml`:
        ```toml
        OPENAI_API_KEY = "sk-your-api-key-here"
        ```
        2. Lub ustaw zmienną środowiskową `OPENAI_API_KEY`
        """)
    
    st.divider()
    
    # Dodawanie nowej uprawy
    st.subheader("Dodaj nową uprawę")
    
    with st.form("dodaj_uprawe"):
        nazwa_uprawy = st.text_input("Nazwa uprawy:")
        liczba_zadan = st.number_input("Liczba zadań:", min_value=1, max_value=10, value=1)
        
        zadania_nowej_uprawy = []
        for i in range(int(liczba_zadan)):
            st.write(f"**Zadanie {i+1}:**")
            col1, col2 = st.columns([1, 2])
            with col1:
                data_zadania = st.date_input(f"Data zadania {i+1}:", key=f"data_{i}")
            with col2:
                opis_zadania = st.text_input(f"Opis zadania {i+1}:", key=f"opis_{i}")
            
            if data_zadania and opis_zadania:
                zadania_nowej_uprawy.append({
                    'data': data_zadania.strftime('%Y-%m-%d'),
                    'opis': opis_zadania
                })
        
        if st.form_submit_button("Dodaj uprawę"):
            if nazwa_uprawy and zadania_nowej_uprawy:
                uprawa_id = nazwa_uprawy.lower().replace(' ', '_').replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
                
                # Dodaj do bazy
                dodaj_uprawe_do_bazy(client, uprawa_id, {
                    'nazwa': nazwa_uprawy,
                    'zadania': zadania_nowej_uprawy
                })
                
                # Dodaj do wybranych upraw
                wybrane_uprawy.append(uprawa_id)
                zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy)
                
                st.success(f"Uprawa '{nazwa_uprawy}' została dodana do bazy!")
                st.rerun()
            else:
                st.error("Wypełnij wszystkie pola!")
    
    st.divider()
    
    # Usuwanie upraw
    st.subheader("Usuń uprawę")
    if uprawy:
        uprawa_do_usuniecia = st.selectbox(
            "Wybierz uprawę do usunięcia:",
            options=list(uprawy.keys()),
            format_func=lambda x: uprawy[x]['nazwa']
        )
        
        if st.button("🗑️ Usuń uprawę", type="secondary"):
            if uprawa_do_usuniecia:
                usun_uprawe_z_bazy(client, uprawa_do_usuniecia)
                if uprawa_do_usuniecia in wybrane_uprawy:
                    wybrane_uprawy.remove(uprawa_do_usuniecia)
                    zapisz_wybrane_uprawy_do_bazy(client, wybrane_uprawy)
                st.success(f"Uprawa '{uprawy[uprawa_do_usuniecia]['nazwa']}' została usunięta!")
                st.rerun()

# Domyślny widok
if 'main_view' not in st.session_state:
    st.session_state['main_view'] = 'kalendarz'

if st.session_state['main_view'] == 'kalendarz':
    col1, col2 = st.columns([3, 1])

    context_day = None
    context_action = None
    if 'context_day' in st.session_state:
        context_day = st.session_state['context_day']
    if 'context_action' in st.session_state:
        context_action = st.session_state['context_action']

    with col1:
        st.header("Kalendarz")
        dzis = datetime.now()
        col_miesiac, col_rok = st.columns(2)
        with col_miesiac:
            miesiac = st.selectbox("Miesiąc:", range(1, 13), 
                                  index=dzis.month-1, 
                                  format_func=lambda x: str(calendar.month_name[x]))
        with col_rok:
            rok = st.selectbox("Rok:", range(2024, 2027), 
                              index=2025-2024 if dzis.year >= 2025 else 0)
        if uprawy:
            rysuj_kalendarz(rok, miesiac, uprawy, wybrane_uprawy)
        else:
            st.info("Brak upraw w bazie danych. Dodaj pierwszą uprawę w panelu bocznym.")

        # MENU OPCJI I FORMULARZE POD KALENDARZEM
        if context_day and context_action == 'menu':
            st.markdown(f"### Opcje dla dnia {context_day}")
            zadania_w_dniu = []
            for uprawa_id, uprawa in uprawy.items():
                for zad in uprawa['zadania']:
                    if zad['data'] == context_day:
                        zadania_w_dniu.append((uprawa_id, uprawa['nazwa'], zad['opis']))
            col_add, col_del, col_close = st.columns([2,2,1])
            dodaj = col_add.button("➕ Dodaj wydarzenie")
            usun = False
            if zadania_w_dniu:
                usun = col_del.button("🗑️ Usuń wydarzenie")
            zamknij = col_close.button("Zamknij")
            if dodaj:
                st.session_state['context_action'] = 'add'
            elif usun:
                st.session_state['context_action'] = 'remove'
            elif zamknij:
                st.session_state['context_action'] = None
                st.session_state['context_day'] = None

        if context_day and context_action == 'add':
            st.markdown(f"### ➕ Dodaj wydarzenie na {context_day}")
            with st.form("add_event_form_main"):
                uprawa_options = list(uprawy.keys())
                uprawa_nazwa = st.selectbox("Uprawa:", [uprawy[u]['nazwa'] for u in uprawa_options], key=f"uprawa_add_{context_day}")
                opis = st.text_input("Opis zadania:", key=f"opis_add_{context_day}")
                submitted = st.form_submit_button("Dodaj")
                if submitted and opis and uprawa_nazwa:
                    uprawa_id = [u for u in uprawa_options if uprawy[u]['nazwa'] == uprawa_nazwa][0]
                    uprawa = uprawy[uprawa_id]
                    uprawa['zadania'].append({'data': context_day, 'opis': opis})
                    dodaj_uprawe_do_bazy(client, uprawa_id, uprawa)
                    st.success("Dodano wydarzenie!")
                    st.session_state['context_action'] = None
                    st.session_state['context_day'] = None
                    st.rerun()

        if context_day and context_action == 'remove':
            st.markdown(f"### 🗑️ Usuń wydarzenie z {context_day}")
            zadania_do_usuniecia = []
            for uprawa_id, uprawa in uprawy.items():
                for i, zad in enumerate(uprawa['zadania']):
                    if zad['data'] == context_day:
                        zadania_do_usuniecia.append((uprawa_id, i, uprawa['nazwa'], zad['opis']))
            if zadania_do_usuniecia:
                with st.form("remove_event_form_main"):
                    idx = st.selectbox("Wybierz zadanie do usunięcia:", list(range(len(zadania_do_usuniecia))), format_func=lambda i: f"{zadania_do_usuniecia[i][2]}: {zadania_do_usuniecia[i][3]}")
                    submitted = st.form_submit_button("Usuń")
                    if submitted:
                        uprawa_id, i, _, _ = zadania_do_usuniecia[idx]
                        del uprawy[uprawa_id]['zadania'][i]
                        dodaj_uprawe_do_bazy(client, uprawa_id, uprawy[uprawa_id])
                        st.success("Usunięto wydarzenie!")
                        st.session_state['context_action'] = None
                        st.session_state['context_day'] = None
                        st.rerun()
            else:
                st.info("Brak wydarzeń do usunięcia na ten dzień.")
                st.session_state['context_action'] = None
                st.session_state['context_day'] = None

    with col2:
        st.header("Zadania")
        # Zakładki: Wskazany dzień / Dzisiaj
        tab1, tab2 = st.tabs(["Wskazany dzień", "Dzisiaj"])
        with tab1:
            selected_day = st.session_state.get('selected_day')
            if selected_day:
                data_selected = datetime.strptime(selected_day, '%Y-%m-%d').date()
                st.subheader(f"Zadania na {data_selected.strftime('%d.%m.%Y')}")
                zadania_selected = pobierz_zadania_na_dzien(data_selected, uprawy, wybrane_uprawy)
                if zadania_selected:
                    for zadanie in zadania_selected:
                        st.info(f"**{zadanie['uprawa']}**: {zadanie['opis']}")
                else:
                    st.write("Brak zadań na ten dzień")
            else:
                st.write("Kliknij dzień w kalendarzu, aby zobaczyć zadania.")
        with tab2:
            st.subheader("Dzisiaj")
            if uprawy:
                zadania_dzis = pobierz_zadania_na_dzien(dzis, uprawy, wybrane_uprawy)
                if zadania_dzis:
                    for zadanie in zadania_dzis:
                        st.info(f"**{zadanie['uprawa']}**: {zadanie['opis']}")
                else:
                    st.write("Brak zadań na dzisiaj")
            else:
                st.write("Brak upraw w bazie")
        st.divider()
        # Zadania na następny tydzień
        st.subheader("Następny tydzień")
        if uprawy:
            jutro = dzis.date() + timedelta(days=1)
            za_tydzien = jutro + timedelta(days=7)
            zadania_tydzien = pobierz_zadania_w_zakresie(jutro, za_tydzien, uprawy, wybrane_uprawy)
            if zadania_tydzien:
                for zadanie in zadania_tydzien:
                    st.info(f"**{zadanie['data'].strftime('%d.%m')}** - {zadanie['uprawa']}: {zadanie['opis']}")
            else:
                st.write("Brak zadań na następny tydzień")
        else:
            st.write("Brak upraw w bazie")
    st.divider()
    col1b, col2b = st.columns(2)
    with col1b:
        st.caption("🌱 Kalendarz Ogrodnika - Planuj swoje uprawy z łatwością!")
    with col2b:
        if uprawy:
            st.caption(f"📊 Liczba upraw w bazie: {len(uprawy)}")
        else:
            st.caption("📊 Baza danych jest pusta")
else:
    st.title("Zarządzaj uprawami")
    if not uprawy:
        st.info("Brak upraw w bazie.")
    else:
        uprawa_ids = list(uprawy.keys())
        uprawa_nazwy = [uprawy[u]['nazwa'] for u in uprawa_ids]
        idx = st.selectbox("Wybierz uprawę do edycji:", range(len(uprawa_ids)), format_func=lambda i: uprawa_nazwy[i], key="select_uprawa_edit")
        uprawa_id = uprawa_ids[idx]
        uprawa = uprawy[uprawa_id]

        # Edycja nazwy uprawy
        new_nazwa = st.text_input("Nazwa uprawy:", value=uprawa['nazwa'], key=f"nazwa_{uprawa_id}")
        # Edycja emoji koloru uprawy
        EMOJI_KOLORY = [
            ("🟩", "Zielony"),
            ("🟥", "Czerwony"),
            ("🟦", "Niebieski"),
            ("🟨", "Żółty"),
            ("🟧", "Pomarańczowy"),
            ("🟪", "Fioletowy"),
            ("⬛", "Czarny"),
            ("⬜", "Biały")
        ]
        emoji_values = [e[0] for e in EMOJI_KOLORY]
        emoji_dict = dict(EMOJI_KOLORY)
        emoji_default = uprawa.get('emoji', '🟩')
        new_emoji_val = st.selectbox(
            "Kolor uprawy:",
            options=emoji_values,
            index=emoji_values.index(emoji_default) if emoji_default in emoji_values else 0,
            format_func=lambda e: f"{e} {emoji_dict[e]}",
            key=f"emoji_{uprawa_id}"
        )

        # Edycja i usuwanie zadań
        st.markdown("#### Zadania")
        zadania = uprawa['zadania']
        zadania_to_remove = []
        for i, zad in enumerate(zadania):
            col1, col2, col3, col4 = st.columns([2,4,2,1])
            with col1:
                new_data = st.date_input("Data", value=pd.to_datetime(zad['data']), key=f"data_{uprawa_id}_{i}")
            with col2:
                new_opis = st.text_input("Opis", value=zad['opis'], key=f"opis_{uprawa_id}_{i}")
            with col3:
                checked = zad.get('zrealizowane', False)
                new_checked = st.checkbox("Zrealizowane?", value=checked, key=f"done_{uprawa_id}_{i}")
            with col4:
                if st.button("Usuń", key=f"del_{uprawa_id}_{i}"):
                    zadania_to_remove.append(i)
            # Aktualizuj zadanie jeśli zmieniono
            zad['data'] = new_data.strftime('%Y-%m-%d')
            zad['opis'] = new_opis
            zad['zrealizowane'] = new_checked
        # Usuwanie wybranych zadań
        for i in sorted(zadania_to_remove, reverse=True):
            del zadania[i]

        # Dodawanie nowego zadania
        st.markdown("#### Dodaj nowe zadanie")
        with st.form(f"add_zadanie_{uprawa_id}"):
            new_data = st.date_input("Data zadania", key=f"add_data_{uprawa_id}")
            new_opis = st.text_input("Opis zadania", key=f"add_opis_{uprawa_id}")
            add_submit = st.form_submit_button("Dodaj zadanie")
            if add_submit and new_opis:
                zadania.append({'data': new_data.strftime('%Y-%m-%d'), 'opis': new_opis, 'zrealizowane': False})
                dodaj_uprawe_do_bazy(client, uprawa_id, {'nazwa': new_nazwa, 'zadania': zadania, 'emoji': new_emoji_val})
                st.success("Dodano zadanie!")
                st.rerun()

        # Zapisz zmiany
        if st.button("Zapisz zmiany", key=f"save_{uprawa_id}"):
            # ZAWSZE zapisuj emoji!
            dodaj_uprawe_do_bazy(client, uprawa_id, {'nazwa': new_nazwa, 'zadania': zadania, 'emoji': new_emoji_val})
            st.success("Zapisano zmiany!")
            st.rerun()
