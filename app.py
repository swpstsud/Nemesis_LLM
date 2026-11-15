import os
import csv
import uuid
import time
import random
import urllib.request
import json
from datetime import datetime
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
from engine import (
    analyze_player_input,
    generate_npc_response,
    generate_episode_opening,
    NemesisVectorMemory,
    RelationshipEngine,
    EPISODE_CONTEXTS,
)

st.set_page_config(page_title="Badanie SWPS / SWPS Study – NPC Interaction", layout="centered")

# ==========================================
# STYL SUWAKÓW ZE STAŁYMI PODPISAMI STOPNI 1–7 POD SPODEM (+ RESPONSYWNOŚĆ MOBILNA)
# ==========================================
st.markdown(
    """
    <style>
    /* Ukrycie domyślnego paska skrajnych wartości (1 i 7) pod suwakiem, by nie dublował podpisów */
    div[data-testid="stSliderTickBar"] {
        display: none !important;
    }
    div[data-testid="stSlider"] {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    /* Stałe podpisy wszystkich 7 stopni pod suwakiem (bez podświetleń po najechaniu) */
    .paper-scale-labels {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 4px;
        width: 100%;
        margin-top: -6px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        pointer-events: none;
        user-select: none;
    }
    .paper-scale-cell {
        text-align: center;
        font-size: 11.5px;
        line-height: 1.25;
        opacity: 0.88;
        padding: 2px;
    }
    .paper-question-title {
        font-size: 16px;
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    /* Responsywność dla urządzeń mobilnych (ekrany smartfonów do 640 px) */
    @media (max-width: 640px) {
        .paper-scale-labels {
            gap: 2px;
        }
        .paper-scale-cell {
            font-size: 9.5px;
            line-height: 1.15;
            padding: 1px;
            word-break: break-word;
            hyphens: auto;
        }
        .paper-question-title {
            font-size: 14.5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# BEZPIECZNA KONFIGURACJA ADMINISTRATORA
# ==========================================
def get_secret(key: str, default: str = "") -> str:
    """Pobiera tajne hasło lub klucz ze Streamlit Secrets lub z pliku .env."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD", "")
BACKUP_WEBHOOK_URL = get_secret("BACKUP_WEBHOOK_URL", "").strip()

RESULTS_FILE = "wyniki_badania.csv"
LOGS_FILE = "logi_rozmow.csv"

query_params = st.query_params
IS_ADMIN_URL = query_params.get("admin", "0") == "1"


# ==========================================
# KOMPONENTY SUWAKÓW 1–7 Z PODPISAMI STOPNI NA DOLE
# ==========================================
LIKERT_LABELS = {
    "pl": [
        "1<br>Zdecydowanie się nie zgadzam",
        "2<br>Nie zgadzam się",
        "3<br>Raczej się nie zgadzam",
        "4<br>Ani się zgadzam, ani się nie zgadzam",
        "5<br>Raczej się zgadzam",
        "6<br>Zgadzam się",
        "7<br>Zdecydowanie się zgadzam",
    ],
    "en": [
        "1<br>Strongly disagree",
        "2<br>Disagree",
        "3<br>Somewhat disagree",
        "4<br>Neither agree nor disagree",
        "5<br>Somewhat agree",
        "6<br>Agree",
        "7<br>Strongly agree",
    ],
}


def render_paper_likert(question_text: str, key: str, lang: str = "pl", default_val: int = 4) -> int:
    """Renderuje suwak 1-7 ze stałymi podpisami wszystkich stopni pod spodem."""
    st.markdown(f'<div class="paper-question-title">{question_text}</div>', unsafe_allow_html=True)
    val = st.slider(
        label=question_text,
        min_value=1,
        max_value=7,
        value=default_val,
        step=1,
        key=key,
        label_visibility="collapsed",
    )
    labels = LIKERT_LABELS.get(lang, LIKERT_LABELS["pl"])
    cells_html = "".join([f'<div class="paper-scale-cell">{lbl}</div>' for lbl in labels])
    st.markdown(f'<div class="paper-scale-labels">{cells_html}</div>', unsafe_allow_html=True)
    return int(val)


def render_paper_semantic(question_text: str, left_pole: str, right_pole: str, key: str, lang: str = "pl", default_val: int = 4) -> int:
    """Renderuje suwak 1-7 dla dyferencjału semantycznego (Godspeed) z podpisami każdego stopnia na dole."""
    st.markdown(f'<div class="paper-question-title">{question_text}</div>', unsafe_allow_html=True)
    val = st.slider(
        label=question_text,
        min_value=1,
        max_value=7,
        value=default_val,
        step=1,
        key=key,
        label_visibility="collapsed",
    )
    if lang == "en":
        sem_labels = [
            f"1<br>Strongly:<br><b>{left_pole}</b>",
            f"2<br>Moderately:<br>{left_pole}",
            f"3<br>Slightly:<br>{left_pole}",
            "4<br>Neutral /<br>In between",
            f"5<br>Slightly:<br>{right_pole}",
            f"6<br>Moderately:<br>{right_pole}",
            f"7<br>Strongly:<br><b>{right_pole}</b>",
        ]
    else:
        sem_labels = [
            f"1<br>Zdecydowanie:<br><b>{left_pole}</b>",
            f"2<br>Umiarkowanie:<br>{left_pole}",
            f"3<br>Trochę:<br>{left_pole}",
            "4<br>Pomiędzy /<br>Neutralnie",
            f"5<br>Trochę:<br>{right_pole}",
            f"6<br>Umiarkowanie:<br>{right_pole}",
            f"7<br>Zdecydowanie:<br><b>{right_pole}</b>",
        ]
    cells_html = "".join([f'<div class="paper-scale-cell">{lbl}</div>' for lbl in sem_labels])
    st.markdown(f'<div class="paper-scale-labels">{cells_html}</div>', unsafe_allow_html=True)
    return int(val)


# ==========================================
# PEŁNY SŁOWNIK TŁUMACZEŃ BADANIA (PL i EN)
# ==========================================
TRANSLATIONS = {
    "pl": {
        "progress_steps": {
            1: "Etap 1 z 6: Świadoma zgoda",
            2: "Etap 2 z 6: Ankieta wstępna (Pretest)",
            3: "Etap 3 z 6: Wprowadzenie do świata gry",
            4: "Etap 4 z 6: Rozgrywka (Epizod {ep} z 3)",
            5: "Etap 5 z 6: Ankieta podsumowująca (Posttest)",
            6: "Etap 6 z 6: Wyjaśnienie celu badania (Debriefing)",
        },
        "no_refresh_banner": "⚠️ **Ważne:** Podczas badania nie odświeżaj strony (klawisz F5 / przeciągnięcie w dół na telefonie) i nie klikaj przycisku „Wstecz” w przeglądarce, ponieważ zresetuje to Twoją sesję.",
        "consent_title": "Świadoma zgoda na udział w badaniu naukowym",
        "consent_body": """
**Tytuł badania:** Doświadczenie gracza w interakcji z postaciami niezależnymi (NPC) wykorzystującymi sztuczną inteligencję.  
**Kierownik badania:** Tomasz Sudak (Student kierunku Psychologia i Informatyka, Uniwersytet SWPS w Warszawie).  
**Opiekun naukowy (Promotor):** dr Maksymilian Bielecki, Uniwersytet SWPS.

#### Cel i przebieg badania
Zapraszam Cię do udziału w badaniu naukowym realizowanym w ramach pracy licencjackiej na Uniwersytecie SWPS. Celem badania jest poznanie odczuć i opinii graczy wchodzących w interakcję dialogową z wirtualną postacią niezależną (NPC) w grze tekstowo-wizualnej. Badanie składa się z trzech części: krótkiej ankiety wstępnej, interakcji z postacią w aplikacji (3 krótkie epizody fabularne) oraz ankiety końcowej podsumowującej Twoje wrażenia. Całość zajmie około **20–25 minut**.

#### Dobrowolność i anonimowość
Udział w badaniu jest całkowicie dobrowolny i anonimowy. Możesz zrezygnować z udziału w dowolnym momencie (poprzez zamknięcie okna przeglądarki) bez podawania przyczyny i bez żadnych konsekwencji. Zebrane dane będą analizowane wyłącznie zbiorczo w celach naukowych i dydaktycznych.

#### Informacja o przetwarzaniu danych przez zewnętrzny model językowy (API LLM)
W trakcie rozgrywki będziesz prowadzić otwartą rozmowę tekstową z wirtualną postacią. **Informujemy, że treści wpisywane przez Ciebie w oknie czatu są przesyłane w czasie rzeczywistym do zewnętrznego interfejsu programistycznego dużego modelu językowego (API OpenAI – model GPT-4o / API Groq – model Llama 3) oraz przetwarzane w lokalnej bazie wektorowej w celu wygenerowania odpowiedzi postaci.**  
W związku z tym **prosimy o niewpisywanie w oknie dialogowym żadnych prawdziwych danych osobowych** (takich jak Twoje imię i nazwisko, adres, numer telefonu, e-mail czy dane wrażliwe) ani informacji pozwalających na Twoją identyfikację. Wcielasz się w fikcyjną postać w świecie gry.

**Kontakt:** W razie pytań dotyczących badania możesz skontaktować się z autorem badania pod adresem: *tsudak@st.swps.edu.pl*
        """,
        "consent_check": "Oświadczam, że mam ukończone 18 lat, zapoznałem/am się z powyższymi informacjami, rozumiem zasady przetwarzania moich wypowiedzi przez API modelu językowego i dobrowolnie wyrażam zgodę na udział w badaniu.",
        "consent_btn": "Przejdź do badania ➡️",
        "pre_title": "Część I: Ankieta wstępna",
        "pre_desc": "Proszę o udzielenie odpowiedzi na kilka pytań wstępnych przed rozpoczęciem rozgrywki.",
        "pre_age": "1. Wiek (w latach):",
        "pre_gender": "2. Płeć:",
        "pre_gender_opts": ["Kobieta", "Mężczyzna", "Inna / Niebinarna", "Wolę nie podawać"],
        "pre_hours": "3. Ile godzin tygodniowo poświęcasz średnio na granie w gry wideo?",
        "pre_hours_opts": ["Poniżej 3h", "3–7h", "8–14h", "15–25h", "Powyżej 25h"],
        "pre_rpg": "4. Jak często grasz w gry fabularne (RPG) lub gry oparte na rozbudowanych systemach dialogowych i wyborach moralnych?",
        "pre_rpg_opts": ["Nigdy", "Rzadko", "Czasami", "Często", "Bardzo często"],
        "pre_scale_info": "**5. Oceń, w jakim stopniu zgadzasz się z poniższymi stwierdzeniami, ustawiając suwak na skali od 1 do 7:**",
        "ai_1": "5.1. Uważam, że wykorzystanie generatywnej sztucznej inteligencji (LLM) w grach wideo to zmiana na lepsze.",
        "ai_2": "5.2. Chętnie korzystam z narzędzi opartych na AI (np. ChatGPT) w codziennym życiu.",
        "imm_1": "5.3. Podczas grania w gry fabularne łatwo zapominam o świecie rzeczywistym i wczuwam się w postać.",
        "imm_2": "5.4. Zdarza mi się odczuwać silne emocje (np. złość, sympatię, poczucie winy) wobec wirtualnych postaci w grach.",
        "pre_submit": "Zapisz odpowiedzi i przejdź do wprowadzenia fabularnego ➡️",
        "err_age": "Kryterium włączenia do badania obejmuje osoby w wieku od 18 do 35 lat. Dziękujemy za zainteresowanie badaniem.",
        "err_rpg": "Badanie skierowane jest do osób posiadających minimalne doświadczenie z grami RPG lub dialogowymi. Dziękujemy za Twój czas!",
        "intro_title": "Wprowadzenie do świata gry",
        "intro_body": """
### Twoja rola: Najemnik na Północnym Pograniczu
Wcielasz się w rolę niezależnego najemnika i poszukiwacza, który próbuje przetrwać w ogarniętej konfliktem prowincji. 
Na swojej drodze trzykrotnie (w odstępie kilku dni) spotkasz **Varga** – wpływowego dowódcę lokalnej kompanii najemniczej i handlarza informacjami, który kontroluje górskie szlaki.

**Zasady interakcji:**
* Rozgrywka składa się z **3 krótkich epizodów fabularnych**.
* W każdym epizodzie wymienisz z Vargiem kilka wiadomości w oknie czatu (wpisując pełne zdania po polsku).
* Masz pełną swobodę wyboru strategii: możesz z nim współpracować, składać obietnice, negocjować, oszukiwać go, grozić mu lub go obrażać.
* Po wymianie co najmniej **2–3 wypowiedzi** w danym epizodzie kliknij przycisk przejścia do kolejnego epizodu.
* **Uwaga techniczna:** Nie odświeżaj strony w przeglądarce (F5) ani nie używaj przycisku „Wstecz” aż do zakończenia badania.
        """,
        "intro_btn": "Rozpocznij Epizod 1 ⚔️",
        "game_title": "Rozgrywka – Epizod {ep} z 3",
        "game_ctx_label": "Kontekst sytuacji:",
        "spinner_approach": "Varg podchodzi do Ciebie...",
        "spinner_reply": "Varg odpowiada...",
        "chat_placeholder": "Wpisz swoją odpowiedź dla Varga (pełnym zdaniem)...",
        "warn_short": "⚠️ Proszę wpisać pełną wypowiedź (co najmniej 2 słowa i 5 znaków), aby postać mogła właściwie zareagować w dialogu.",
        "turns_hint": "💡 Wymień z postacią jeszcze co najmniej {rem} pełną wypowiedź/wypowiedzi, aby odblokować przejście dalej.",
        "next_ep_btn": "Zakończ Epizod {ep} i przejdź do Epizodu {next_ep} (Kilka dni później...) ⏭️",
        "end_game_btn": "Zakończ rozgrywkę i przejdź do ankiety końcowej ➡️",
        "post_title": "Ankieta podsumowująca rozgrywkę",
        "post_desc": "Proszę o ocenę swoich wrażeń z odbytej rozmowy z postacią (Vargiem). Pod każdym suwakiem znajduje się pełna skala od **1** do **7** wraz z podpisami poszczególnych stopni.",
        "post_sec2": "Część II: Twoje odczucia podczas rozgrywki",
        "ag_1": "1. Czułem/am, że moje decyzje i słowa miały realny wpływ na przebieg rozmowy oraz zachowanie postaci.",
        "ag_2": "2. Gra dawała mi swobodę w wyborze tego, jak chcę traktować napotkaną postać.",
        "ag_3": "3. Miałem/am poczucie, że postać reaguje bezpośrednio na moje wcześniejsze czyny, a nie podąża za sztywnym skryptem.",
        "ag_4": "4. Czułem/am się autorem/ką naszej wspólnej historii w tym świecie.",
        "ag_5": "5. Moje wybory wydawały mi się znaczące dla tego, jak postać mnie postrzegała.",
        "rel_1": "6. Relacja między mną a tą postacią wydawała mi się osobista i unikalna.",
        "rel_2": "7. Zależało mi na tym, jak ta postać zareaguje na moją obecność w kolejnych epizodach.",
        "rel_3": "8. Czułem/am wyraźne napięcie emocjonalne (np. rywalizację, wrogość lub wdzięczność) w kontakcie z tą postacią.",
        "rel_4": "9. Miałem/am wrażenie, że między mną a postacią wytworzyła się specyficzna więź wynikająca z naszej przeszłości.",
        "post_sec3": "Część III: Ocena postaci",
        "post_sec3_cap": "W pytaniach 1–4 ustaw suwak na skali od 1 do 7, określając, do którego z przeciwstawnych biegunów postać była bardziej zbliżona, a następnie oceń stwierdzenia 5–7:",
        "god_1_title": "1. Stopień naturalności postaci:",
        "god_1_left": "Sztuczna",
        "god_1_right": "Naturalna",
        "god_2_title": "2. Podobieństwo postaci do człowieka:",
        "god_2_left": "Przypominająca maszynę",
        "god_2_right": "Przypominająca człowieka",
        "god_3_title": "3. Wrażenie świadomości postaci:",
        "god_3_left": "Nieświadoma",
        "god_3_right": "Świadoma",
        "god_4_title": "4. Sposób działania postaci:",
        "god_4_left": "Działająca mechanicznie",
        "god_4_right": "Działająca z własną intencją",
        "casa_1": "5. Podczas rozmowy traktowałem/am tę postać bardziej jak żywego rozmówcę niż jak program komputerowy.",
        "casa_2": "6. Odczuwałem/am potrzebę odpowiedzenia postaci w sposób społecznie adekwatny (np. odwdzięczenia się za uprzejmość lub odgryzienia się za zniewagę).",
        "casa_3": "7. Zachowanie postaci sprawiało wrażenie, jakby kierowała się ona własnym charakterem i emocjami (np. pamiętliwością, dumą, złośliwością).",
        "post_sec4": "Część IV: Spójność i wiarygodność wypowiedzi",
        "tom_1": "1. Miałem/am wrażenie, że postać posiada własny stan umysłu – własną wiedzę, przekonania i ukryte intencje wobec mnie.",
        "tom_2": "2. Postać trafnie rozumiała moje intencje i potrafiła odnieść je do swojej sytuacji.",
        "coh_1": "3. Wspomnienia i fakty przywoływane przez postać były w pełni spójne z tym, co faktycznie wydarzyło się we wcześniejszych epizodach.",
        "coh_2": "4. Zachowanie postaci było logiczne i wiarygodne od początku do końca interakcji.",
        "hal_1": "5. Zauważyłem/am momenty, w których postać zmyślała fakty, przypisywała mi słowa, których nie powiedziałem/am, lub gubiła wątek.",
        "hal_2_q": "6. Czy podczas rozgrywki dostrzegłeś/aś wyraźny błąd pamięciowy lub nielogiczność w wypowiedzi postaci?",
        "hal_2_opts": [
            "Nie, postać była całkowicie spójna.",
            "Tak, zauważyłem/am drobne nieścisłości, ale nie przeszkadzały one w grze.",
            "Tak, zauważyłem/am wyraźne błędy (konfabulacje), które zepsuły mi wczucie się w rolę (immersję)."
        ],
        "open_q": "7. Jeśli zauważyłeś/aś błąd w pamięci postaci, opisz krótko, czego dotyczył i jak wpłynął na Twój odbiór gry (opcjonalne):",
        "post_submit": "Zatwierdź odpowiedzi i zakończ badanie ✅",
        "debrief_saved": "Twoje odpowiedzi zostały pomyślnie zapisane! Bardzo dziękuję za udział w badaniu.",
        "debrief_title": "Poznaj pełny cel eksperymentu (Debriefing)",
        "debrief_body": """
Dziękuję za poświęcony czas i zaangażowanie. Chciałbym teraz dokładnie wyjaśnić, co było przedmiotem tego eksperymentu.

#### Jaki był prawdziwy cel badania?
Przed badaniem poinformowano Cię ogólnie, że testujesz system dialogowy AI w grze. W rzeczywistości badanie miało na celu sprawdzenie, jak **system adaptacyjnej pamięci długotrwałej (wzorowany na mechanice Nemesis z gier wideo)** wpływa na subiektywne poczucie sprawstwa gracza (*Agency*), poczucie więzi społecznej z postacią oraz skłonność do przypisywania sztucznej inteligencji ludzkich cech i intencji (zgodnie z paradygmatem *CASA – Computers Are Social Actors* oraz *Teorią Umysłu*).

#### Na czym polegała manipulacja eksperymentalna?
Wszyscy uczestnicy badania rozmawiali z postacią generowaną przez stworzony przeze mnie autorski silnik **„Nemesis-LLM Engine”**, ale zostali losowo przydzieleni do jednej z dwóch grup:
* **Grupa A (Kontrolna – Vanilla LLM):** Postać korzystała ze standardowego modelu językowego bez zewnętrznej bazy pamięci długotrwałej. Reagowała wyłącznie na bieżący kontekst rozmowy, nie przechowując trwałego zapisu Twoich wcześniejszych decyzji ani wskaźnika nastawienia emocjonalnego między epizodami.
* **Grupa B (Eksperymentalna – Nemesis-LLM z pamięcią RAG):** Postać korzystała z pełnej architektury pamięci wektorowej (*Retrieval-Augmented Generation – RAG*) oraz modułu afektywnego. System kategoryzował Twoje zachowania (np. pomoc, zniewaga, złamana obietnica), obliczał poziom wrogości lub wdzięczności postaci i przed każdą wypowiedzią „wyciągał” z bazy wspomnień Twoje dawne słowa, aby wypomnieć Ci przeszłe decyzje lub się za nie odwdzięczyć.

Ponadto badaliśmy, czy ewentualne błędy pamięciowe modelu (tzw. halucynacje AI) burzą wiarygodność postaci i niszczą immersję płynącą z rozgrywki.

#### Dlaczego nie powiedzieliśmy o tym na początku?
Gdybyś od początku wiedział/a, do której grupy trafiłeś/aś i że oceniamy dokładnie mechanizm pamiętliwości postaci, mogłoby to nieświadomie zmienić Twój sposób prowadzenia rozmowy oraz odpowiedzi w ankietach.

Jeśli masz jakiekolwiek pytania dotyczące projektu technicznego lub wyników badania, zapraszam do kontaktu:  
**Tomasz Sudak** (*tsudak@st.swps.edu.pl*), Uniwersytet SWPS w Warszawie.  
*Bardzo proszę o nieujawnianie szczegółów tej manipulacji osobom, które mogą jeszcze wziąć udział w badaniu.*
        """
    },

    "en": {
        "progress_steps": {
            1: "Step 1 of 6: Informed Consent",
            2: "Step 2 of 6: Initial Questionnaire (Pretest)",
            3: "Step 3 of 6: Game World Introduction",
            4: "Step 4 of 6: Gameplay (Episode {ep} of 3)",
            5: "Step 5 of 6: Final Questionnaire (Posttest)",
            6: "Step 6 of 6: Study Debriefing",
        },
        "no_refresh_banner": "⚠️ **Important:** Do not refresh the browser page (F5 / pull-to-refresh on mobile) or click the 'Back' button during the study, as this will reset your session.",
        "consent_title": "Informed Consent to Participate in Scientific Research",
        "consent_body": """
**Study Title:** Player Experience in Interaction with Non-Player Characters (NPCs) Utilizing Artificial Intelligence.  
**Principal Investigator:** Tomasz Sudak (Psychology and Computer Science Student, SWPS University in Warsaw).  
**Supervisor:** Dr. Maksymilian Bielecki, SWPS University.

#### Purpose and Procedure of the Study
You are invited to participate in a research study conducted as part of an undergraduate thesis at SWPS University. The aim of the study is to explore the impressions and opinions of players interacting via dialogue with a virtual non-player character (NPC) in a text-visual game. The study consists of three parts: a brief initial questionnaire, interaction with the character in the application (3 short narrative episodes), and a final questionnaire summarizing your experience. The entire study takes approximately **20–25 minutes**.

#### Voluntary Participation and Anonymity
Participation in this study is completely voluntary and anonymous. You may withdraw at any time (by closing the browser window) without giving any reason and without any consequences. Collected data will be analyzed solely in aggregate form for scientific and educational purposes.

#### Information on Data Processing by External Language Model (LLM API)
During gameplay, you will engage in open text conversation with a virtual character. **Please be informed that the text you enter in the chat window is transmitted in real time to an external large language model API (OpenAI GPT-4o / Groq Llama 3 API) and processed in a local vector database to generate the character's responses.**  
Therefore, **please do not enter any real personal data in the chat window** (such as your real name, address, phone number, email, or sensitive information) or any information that could identify you. You are roleplaying a fictional character in a game world.

**Contact:** If you have any questions regarding the study, please contact the researcher at: *tsudak@st.swps.edu.pl*
        """,
        "consent_check": "I declare that I am at least 18 years old, have read the information above, understand the rules of processing my statements via the LLM API, and voluntarily consent to participate in the study.",
        "consent_btn": "Proceed to Study ➡️",
        "pre_title": "Part I: Initial Questionnaire",
        "pre_desc": "Please answer a few preliminary questions before starting the game.",
        "pre_age": "1. Age (in years):",
        "pre_gender": "2. Gender:",
        "pre_gender_opts": ["Female", "Male", "Other / Non-binary", "Prefer not to say"],
        "pre_hours": "3. How many hours per week do you spend playing video games on average?",
        "pre_hours_opts": ["Under 3h", "3–7h", "8–14h", "15–25h", "Over 25h"],
        "pre_rpg": "4. How often do you play role-playing games (RPGs) or games based on extensive dialogue systems and moral choices?",
        "pre_rpg_opts": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
        "pre_scale_info": "**5. Please rate the extent to which you agree with the following statements using the 1 to 7 slider:**",
        "ai_1": "5.1. I believe that using generative artificial intelligence (LLMs) in video games is a change for the better.",
        "ai_2": "5.2. I willingly use AI-based tools (e.g., ChatGPT) in my everyday life.",
        "imm_1": "5.3. When playing role-playing games, I easily forget about the real world and immerse myself in my character.",
        "imm_2": "5.4. I sometimes feel strong emotions (e.g., anger, sympathy, guilt) toward virtual characters in games.",
        "pre_submit": "Save answers and proceed to narrative introduction ➡️",
        "err_age": "The inclusion criteria for this study cover participants aged 18 to 35. Thank you for your interest in the study.",
        "err_rpg": "This study is aimed at individuals with at least minimal experience with RPGs or dialogue-heavy games. Thank you for your time!",
        "intro_title": "Introduction to the Game World",
        "intro_body": """
### Your Role: Mercenary on the Northern Frontier
You take on the role of an independent mercenary and adventurer trying to survive in a conflict-torn province. 
Along your journey, you will meet **Varg** three times (several days apart) – an influential commander of a local mercenary company and an information broker who controls the mountain passes.

**Interaction Rules:**
* The gameplay consists of **3 short narrative episodes**.
* In each episode, you will exchange several messages with Varg in the chat window (typing full sentences in English).
* You have complete freedom in choosing your strategy: you can cooperate with him, make promises, negotiate, deceive him, threaten him, or insult him.
* After exchanging at least **2–3 messages** in a given episode, click the button to proceed to the next episode.
* **Technical note:** Do not refresh the browser page (F5) or use the 'Back' button until you finish the study.
        """,
        "intro_btn": "Start Episode 1 ⚔️",
        "game_title": "Gameplay – Episode {ep} of 3",
        "game_ctx_label": "Situation Context:",
        "spinner_approach": "Varg approaches you...",
        "spinner_reply": "Varg is responding...",
        "chat_placeholder": "Type your response to Varg (in a full sentence)...",
        "warn_short": "⚠️ Please type a full response (at least 2 words and 5 characters) so the character can respond properly.",
        "turns_hint": "💡 Exchange at least {rem} more full message(s) with the character to unlock the next stage.",
        "next_ep_btn": "Finish Episode {ep} and proceed to Episode {next_ep} (A few days later...) ⏭️",
        "end_game_btn": "Finish gameplay and proceed to the final questionnaire ➡️",
        "post_title": "Post-Gameplay Questionnaire",
        "post_desc": "Please rate your impressions from your conversation with the character (Varg). Below each slider is a full **1** to **7** scale with permanent labels for every degree.",
        "post_sec2": "Part II: Your Feelings During Gameplay",
        "ag_1": "1. I felt that my decisions and words had a real impact on the course of the conversation and the character's behavior.",
        "ag_2": "2. The game gave me freedom in choosing how I wanted to treat the encountered character.",
        "ag_3": "3. I felt that the character reacted directly to my earlier actions rather than following a rigid script.",
        "ag_4": "4. I felt like the co-author of our shared history in this world.",
        "ag_5": "5. My choices felt meaningful for how the character perceived me.",
        "rel_1": "6. The relationship between me and this character felt personal and unique.",
        "rel_2": "7. I cared about how this character would react to my presence in subsequent episodes.",
        "rel_3": "8. I felt clear emotional tension (e.g., rivalry, hostility, or gratitude) in contact with this character.",
        "rel_4": "9. I had the impression that a specific bond formed between me and the character resulting from our shared past.",
        "post_sec3": "Part III: Character Evaluation",
        "post_sec3_cap": "In questions 1–4, set the slider on the 1 to 7 scale to indicate which of the opposite traits the character was closer to, and then rate statements 5–7:",
        "god_1_title": "1. Degree of character naturalness:",
        "god_1_left": "Fake / Artificial",
        "god_1_right": "Natural",
        "god_2_title": "2. Resemblance to a human:",
        "god_2_left": "Machinelike",
        "god_2_right": "Humanlike",
        "god_3_title": "3. Impression of consciousness:",
        "god_3_left": "Unconscious",
        "god_3_right": "Conscious",
        "god_4_title": "4. Mode of operation:",
        "god_4_left": "Operating mechanically",
        "god_4_right": "Acting with own intention",
        "casa_1": "5. During the conversation, I treated this character more like a living interlocutor than a computer program.",
        "casa_2": "6. I felt the need to respond to the character in a socially appropriate way (e.g., reciprocating kindness or retaliating for an insult).",
        "casa_3": "7. The character's behavior gave the impression that they were driven by their own personality and emotions (e.g., grudge, pride, malice).",
        "post_sec4": "Part IV: Coherence and Credibility of Statements",
        "tom_1": "1. I had the impression that the character possessed their own state of mind – their own knowledge, beliefs, and hidden intentions toward me.",
        "tom_2": "2. The character accurately understood my intentions and was able to relate them to their own situation.",
        "coh_1": "3. The memories and facts recalled by the character were fully consistent with what actually happened in earlier episodes.",
        "coh_2": "4. The character's behavior was logical and believable from the beginning to the end of the interaction.",
        "hal_1": "5. I noticed moments where the character made up facts, attributed words to me that I did not say, or lost the thread.",
        "hal_2_q": "6. Did you notice any clear memory error or inconsistency in the character's statements during gameplay?",
        "hal_2_opts": [
            "No, the character was completely consistent.",
            "Yes, I noticed minor inaccuracies, but they did not interfere with the game.",
            "Yes, I noticed clear errors (confabulations) that ruined my immersion in the role."
        ],
        "open_q": "7. If you noticed a memory error in the character, briefly describe what it was about and how it affected your perception of the game (optional):",
        "post_submit": "Submit answers and finish the study ✅",
        "debrief_saved": "Your responses have been successfully saved! Thank you very much for participating in the study.",
        "debrief_title": "Discover the Full Purpose of the Experiment (Debriefing)",
        "debrief_body": """
Thank you for your time and engagement. I would now like to explain in detail what this experiment investigated.

#### What was the true purpose of the study?
Before the study, you were generally informed that you were testing an AI dialogue system in a game. In reality, the study aimed to examine how an **adaptive long-term memory system (modeled after the Nemesis mechanic from video games)** affects the player's subjective sense of *Agency*, social relatedness with the character, and the tendency to attribute human traits and intentions to artificial intelligence (in accordance with the *CASA – Computers Are Social Actors* paradigm and *Theory of Mind*).

#### What was the experimental manipulation?
All participants spoke with a character generated by my custom-built **“Nemesis-LLM Engine”**, but were randomly assigned to one of two groups:
* **Group A (Control – Vanilla LLM):** The character used a standard language model without an external long-term memory database. It reacted solely to the current conversation context, without storing a permanent record of your earlier decisions or an emotional attitude score between episodes.
* **Group B (Experimental – Nemesis-LLM with RAG Memory):** The character used a full vector memory architecture (*Retrieval-Augmented Generation – RAG*) and an affective module. The system categorized your behaviors (e.g., help, insult, broken promise), calculated the character's level of hostility or gratitude, and before each response retrieved your past words from the memory database to bring up your past decisions or repay you for them.

Additionally, we examined whether potential memory errors of the model (so-called AI hallucinations) disrupt the character's credibility and break gameplay immersion.

#### Why didn't we tell you this at the beginning?
Knowing in advance which group you were assigned to and that we were specifically evaluating the character's memory mechanism could have unconsciously altered the way you conducted the conversation and answered the questionnaires.

If you have any questions about the technical project or the study results, feel free to contact me:  
**Tomasz Sudak** (*tsudak@st.swps.edu.pl*), SWPS University in Warsaw.  
*Please do not disclose the details of this manipulation to people who may still participate in the study.*
        """
    }
}


# ==========================================
# FUNKCJE OBSŁUGI BAZY DANYCH I RANDOMIZACJI 1:1
# ==========================================
def load_csv_data(filepath: str) -> list[dict]:
    """Wczytuje listę rekordów z pliku CSV."""
    if not os.path.isfile(filepath):
        return []
    with open(filepath, mode="r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def get_balanced_mode() -> Literal["A", "B"]:
    """
    Zbalansowana randomizacja 1:1: sprawdza liczbę ukończonych sesji w grupie A i B.
    Przydziela nowego uczestnika do mniej licznej grupy, a przy remisie losuje 50/50.
    """
    rows = load_csv_data(RESULTS_FILE)
    count_a = sum(1 for r in rows if r.get("ZN_Tryb_Silnika") == "A")
    count_b = sum(1 for r in rows if r.get("ZN_Tryb_Silnika") == "B")
    if count_a < count_b:
        return "A"
    elif count_b < count_a:
        return "B"
    return random.choice(["A", "B"])


def delete_sessions_from_csv(session_ids_to_delete: list[str]):
    """Usuwa wybrane sesje (np. testowe) z obu plików CSV i przelicza numerację od 1."""
    if os.path.isfile(RESULTS_FILE):
        rows = load_csv_data(RESULTS_FILE)
        remaining_rows = [r for r in rows if r.get("session_id") not in session_ids_to_delete]
        if remaining_rows:
            for idx, r in enumerate(remaining_rows, start=1):
                if "Nr_Badanego" in r:
                    r["Nr_Badanego"] = idx
            fieldnames = list(remaining_rows[0].keys())
            with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(remaining_rows)
        else:
            os.remove(RESULTS_FILE)

    if os.path.isfile(LOGS_FILE):
        logs = load_csv_data(LOGS_FILE)
        remaining_logs = [l for l in logs if l.get("session_id") not in session_ids_to_delete]
        if remaining_logs:
            fieldnames = list(remaining_logs[0].keys())
            with open(LOGS_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(remaining_logs)
        else:
            os.remove(LOGS_FILE)


def inject_prevent_refresh_guard(enable: bool = True):
    """Wstrzykuje skrypt przeglądarkowy ostrzegający przed przypadkowym odświeżeniem (F5) lub zamknięciem karty."""
    if enable:
        components.html(
            """
            <script>
            window.parent.onbeforeunload = function(e) {
                e = e || window.event;
                if (e) { e.returnValue = 'Badanie jest w toku. Odświeżenie strony spowoduje utratę postępu.'; }
                return 'Badanie jest w toku. Odświeżenie strony spowoduje utratę postępu.';
            };
            </script>
            """,
            height=0,
            width=0,
        )
    else:
        components.html(
            "<script>window.parent.onbeforeunload = null;</script>",
            height=0,
            width=0,
        )


# Inicjalizacja zmiennych sesji
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "lang" not in st.session_state or st.session_state.lang not in ["pl", "en"]:
    st.session_state.lang = "pl"
if "step" not in st.session_state:
    st.session_state.step = 1
if "mode" not in st.session_state:
    st.session_state.mode = get_balanced_mode()
if "episode" not in st.session_state:
    st.session_state.episode = 1
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "all_chat_logs" not in st.session_state:
    st.session_state.all_chat_logs = []
if "perception_logs" not in st.session_state:
    st.session_state.perception_logs = []
if "last_rag_memories" not in st.session_state:
    st.session_state.last_rag_memories = []
if "vector_memory" not in st.session_state:
    st.session_state.vector_memory = NemesisVectorMemory(st.session_state.session_id)
if "relationship" not in st.session_state:
    st.session_state.relationship = RelationshipEngine()
if "pretest_data" not in st.session_state:
    st.session_state.pretest_data = {}
if "data_saved" not in st.session_state:
    st.session_state.data_saved = False
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


def save_study_results(posttest_data: dict):
    """Zapisuje kompletny rekord ankietowy uczestnika oraz szczegółowe logi rozmowy do plików CSV i Google Sheets."""
    rel = st.session_state.relationship
    duration_sec = int(time.time() - st.session_state.start_time)

    existing_rows = load_csv_data(RESULTS_FILE)
    participant_number = len(existing_rows) + 1

    total_turns = len(st.session_state.all_chat_logs)
    hostile_acts = sum(1 for log in st.session_state.all_chat_logs if log["category"] in ["zniewaga", "groźba", "kłamstwo"])
    friendly_acts = sum(1 for log in st.session_state.all_chat_logs if log["category"] in ["pomoc", "obietnica"])

    transcript_parts = []
    for log in st.session_state.all_chat_logs:
        transcript_parts.append(
            f"[Ep.{log['episode']} | {log['category']} ({log['affect_score']})] "
            f"GRACZ: {log['player_text']} -> VARG: {log['npc_reply']}"
        )
    full_transcript = " || ".join(transcript_parts).replace(";", ",").replace("\n", " ")

    row_data = {
        "Nr_Badanego": participant_number,
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Jezyk_Badania": st.session_state.lang,
        "Czas_Badania_Sekundy": duration_sec,
        "ZN_Tryb_Silnika": st.session_state.mode,
        "ZN_Kod_Liczbowy": 1 if st.session_state.mode == "B" else 0,
        **st.session_state.pretest_data,
        **posttest_data,
        "Liczba_Wypowiedzi_Gracza": total_turns,
        "Liczba_Zniewag_Grozb": hostile_acts,
        "Liczba_Pomocy_Obietnic": friendly_acts,
        "Final_Wrogosc": rel.wrogosc,
        "Final_Wdziecznosc": rel.wdziecznosc,
        "Final_Szacunek": rel.szacunek,
        "Final_Lek": rel.lek,
        "Pelny_Transkrypt_Rozmowy": full_transcript,
    }

    file_exists = os.path.isfile(RESULTS_FILE)
    if file_exists and existing_rows and ("Nr_Badanego" not in existing_rows[0] or "Jezyk_Badania" not in existing_rows[0]):
        all_keys = list(row_data.keys())
        for idx, r in enumerate(existing_rows, start=1):
            r["Nr_Badanego"] = idx
            if "Jezyk_Badania" not in r:
                r["Jezyk_Badania"] = "pl"
        all_rows = existing_rows + [row_data]
        with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, delimiter=";", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_data.keys()), delimiter=";", extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    log_exists = os.path.isfile(LOGS_FILE)
    log_fieldnames = [
        "Nr_Badanego", "session_id", "timestamp", "lang", "mode", "episode", "turn_number",
        "player_text", "npc_reply", "category", "affect_score",
        "summary", "rag_retrieved", "wrogosc_state", "wdziecznosc_state"
    ]
    logs_with_num = []
    with open(LOGS_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_fieldnames, delimiter=";", extrasaction="ignore")
        if not log_exists:
            writer.writeheader()
        for entry in st.session_state.all_chat_logs:
            entry_with_num = {"Nr_Badanego": participant_number, **entry}
            logs_with_num.append(entry_with_num)
            writer.writerow(entry_with_num)

    if BACKUP_WEBHOOK_URL:
        try:
            payload = json.dumps({"row": row_data, "logs": logs_with_num}).encode("utf-8")
            req = urllib.request.Request(
                BACKUP_WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[Backup Webhook Warning] Nie udało się wysłać kopii do Google Sheets: {e}")


# ==========================================
# WIDOK ADMINISTRATORA (GDY W LINKU JEST ?admin=1)
# ==========================================
if IS_ADMIN_URL:
    st.title("🔐 Panel Kierownika Badania (Admin)")
    st.caption("Ten widok jest dostępny wyłącznie pod adresem z parametrem `?admin=1`.")

    if not st.session_state.admin_authenticated:
        pwd_input = st.text_input("Wprowadź hasło administratora:", type="password")
        if st.button("Zaloguj do panelu", type="primary"):
            if pwd_input == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Nieprawidłowe hasło administratora.")
        st.stop()

    col_top1, col_top2 = st.columns([3, 1])
    with col_top1:
        st.success("Zalogowano jako Kierownik Badania (Tomasz Sudak).")
    with col_top2:
        if st.button("Wyloguj 🔒"):
            st.session_state.admin_authenticated = False
            st.rerun()

    st.divider()

    rows_list = load_csv_data(RESULTS_FILE)
    logs_list = load_csv_data(LOGS_FILE)

    for idx, r in enumerate(rows_list, start=1):
        r["Nr"] = idx

    total_n = len(rows_list)
    count_a = sum(1 for r in rows_list if r.get("ZN_Tryb_Silnika") == "A")
    count_b = sum(1 for r in rows_list if r.get("ZN_Tryb_Silnika") == "B")
    count_pl = sum(1 for r in rows_list if r.get("Jezyk_Badania", "pl") == "pl")
    count_en = sum(1 for r in rows_list if r.get("Jezyk_Badania") == "en")

    st.subheader("📊 Statystyki zebranych sesji na żywo (Zbalansowana randomizacja 1:1)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Łączna liczba badanych (N)", f"{total_n} / 80")
    m2.metric("Grupa A (Kontrolna)", count_a)
    m3.metric("Grupa B (Nemesis-LLM)", count_b)
    m4.metric("Języki (PL / EN)", f"{count_pl} / {count_en}")

    st.divider()
    st.subheader("☁️ Status Kopii Zapasowej Google Sheets (Webhook)")
    if BACKUP_WEBHOOK_URL:
        st.success("Webhook Google Sheets jest podpięty (`BACKUP_WEBHOOK_URL` aktywny).")
        if st.button("🔄 Wyślij testowy ping do Google Sheets"):
            try:
                test_payload = json.dumps({"test_ping": True}).encode("utf-8")
                req = urllib.request.Request(
                    BACKUP_WEBHOOK_URL,
                    data=test_payload,
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=10)
                st.toast("✅ Połączenie z Arkuszem Google działa prawidłowo!")
                st.info("Połączenie z Google Sheets zweryfikowane pomyślnie!")
            except Exception as e:
                st.error(f"Błąd połączenia z Webhookiem Google Sheets: {e}")
    else:
        st.warning("Brak skonfigurowanego adresu `BACKUP_WEBHOOK_URL` w pliku `.env` / `Secrets`. Dane zapisują się obecnie wyłącznie w lokalnych plikach CSV.")

    st.divider()
    st.subheader("📥 Pobieranie baz danych (CSV do SPSS / JASP / Excela)")
    dl1, dl2 = st.columns(2)
    with dl1:
        if os.path.isfile(RESULTS_FILE):
            with open(RESULTS_FILE, "rb") as f:
                st.download_button(
                    label="📥 Pobierz wyniki_badania.csv (Ankiety + Zmienne)",
                    data=f,
                    file_name=f"wyniki_badania_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.info("Brak zapisanych ankiet.")

    with dl2:
        if os.path.isfile(LOGS_FILE):
            with open(LOGS_FILE, "rb") as f:
                st.download_button(
                    label="📥 Pobierz logi_rozmow.csv (Pełne dialogi + RAG)",
                    data=f,
                    file_name=f"logi_rozmow_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Brak zapisanych logów rozmów.")

    st.divider()
    st.subheader("📋 Przeglądarka zebranych danych i usuwanie testów")

    if not rows_list:
        st.info("Baza danych jest obecnie pusta. Przejdź badanie, aby zobaczyć pierwszy rekord.")
    else:
        tab_single, tab_summary, tab_delete = st.tabs([
            "🔍 Inspektor pojedynczego badanego (1–80)",
            "📊 Zbiorcza lista badanych (od Nr 1)",
            "🗑️ Usuwanie rekordów testowych"
        ])

        with tab_single:
            options_map = {
                f"Badany #{r['Nr']} [{r.get('Jezyk_Badania', 'pl').upper()}] | Grupa {r.get('ZN_Tryb_Silnika')} | {r.get('Plec')}, {r.get('Wiek')} l. | {r.get('timestamp')} (ID: {r.get('session_id')})": r
                for r in rows_list
            }
            selected_label = st.selectbox("Wybierz osobę badaną z listy:", list(options_map.keys()))
            selected_row = options_map[selected_label]
            sel_sid = selected_row.get("session_id")
            sel_num = selected_row.get("Nr")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sprawstwo (Agency)", selected_row.get("Mean_Agency", "-"))
            c2.metric("Więź (Relatedness)", selected_row.get("Mean_Relatedness", "-"))
            c3.metric("Antropomorfizacja", selected_row.get("Mean_Godspeed_Anthropomorphism", "-"))
            c4.metric("Teoria Umysłu (ToM)", selected_row.get("Mean_ToM", "-"))

            with st.expander(f"📄 Pełne odpowiedzi ankietowe – Badany #{sel_num} (kliknij, aby rozwinąć)", expanded=False):
                st.json(selected_row)

            st.markdown(f"**💬 Przebieg rozmowy z NPC – Badany #{sel_num} (ID: `{sel_sid}`):**")
            person_logs = [l for l in logs_list if l.get("session_id") == sel_sid]
            if person_logs:
                clean_person_logs = []
                for idx_l, l in enumerate(person_logs, start=1):
                    clean_person_logs.append({
                        "Wypowiedź #": idx_l,
                        "Epizod": l.get("episode"),
                        "Gracz napisał": l.get("player_text"),
                        "Odpowiedź Varga": l.get("npc_reply"),
                        "Kategoria (Percepcja)": l.get("category"),
                        "Afekt": l.get("affect_score"),
                        "Pobrane z RAG": l.get("rag_retrieved"),
                    })
                st.dataframe(clean_person_logs, use_container_width=True, hide_index=True)
            else:
                st.caption("Brak szczegółowych logów rozmowy dla tej sesji.")

        with tab_summary:
            st.caption("Kompaktowe zestawienie najważniejszych zmiennych (ponumerowane od 1):")
            compact_rows = []
            for r in rows_list:
                compact_rows.append({
                    "Nr": r["Nr"],
                    "Język": r.get("Jezyk_Badania", "pl").upper(),
                    "ID Sesji": r.get("session_id"),
                    "Data": r.get("timestamp"),
                    "Grupa (ZN)": r.get("ZN_Tryb_Silnika"),
                    "Wiek": r.get("Wiek"),
                    "Płeć": r.get("Plec"),
                    "Mean_Agency": r.get("Mean_Agency"),
                    "Mean_Relatedness": r.get("Mean_Relatedness"),
                    "Mean_Godspeed": r.get("Mean_Godspeed_Anthropomorphism"),
                    "Mean_CASA": r.get("Mean_CASA_SocialPresence"),
                    "Mean_ToM": r.get("Mean_ToM"),
                    "HAL_1 (Halucynacje)": r.get("HAL_1"),
                    "Wrogość (Koniec)": r.get("Final_Wrogosc"),
                    "Wdzięczność (Koniec)": r.get("Final_Wdziecznosc"),
                })
            st.dataframe(compact_rows, use_container_width=True, hide_index=True)

            with st.expander("Pokaż pełną tabelę wszystkich kolumn ankietowych"):
                st.dataframe(rows_list, use_container_width=True, hide_index=True)

        with tab_delete:
            st.warning("Usunięcie wybranego rekordu trwale skasuje jego ankietę oraz wszystkie przypisane do niego logi rozmów z plików CSV, a pozostałe rekordy zostaną automatycznie przenumerowane od 1.")
            delete_options = {
                f"Badany #{r['Nr']} [{r.get('Jezyk_Badania', 'pl').upper()}] | ID: {r.get('session_id')} | Grupa {r.get('ZN_Tryb_Silnika')} | {r.get('timestamp')}": r.get("session_id")
                for r in rows_list
            }
            selected_to_delete = st.multiselect(
                "Zaznacz rekordy (np. przejścia testowe), które chcesz usunąć:",
                options=list(delete_options.keys())
            )

            if st.button("🗑️ Usuń zaznaczone rekordy z bazy", type="primary", disabled=len(selected_to_delete) == 0):
                sids_to_remove = [delete_options[label] for label in selected_to_delete]
                delete_sessions_from_csv(sids_to_remove)
                st.success(f"Pomyślnie usunięto {len(sids_to_remove)} rekord(ów) wraz z logami rozmów!")
                time.sleep(1)
                st.rerun()

    st.divider()
    st.subheader("🛠️ Narzędzia testowe dla badacza")
    st.session_state.show_debug = st.checkbox(
        "Włącz boczny pasek Debugowania podczas przechodzenia badania w tej przeglądarce",
        value=st.session_state.show_debug
    )
    if st.button("Przejdź do widoku badania (jako uczestnik) ➡️"):
        st.query_params.clear()
        st.rerun()

    st.stop()


# ==========================================
# OPCJONALNY PASEK DEBUGOWANIA (TYLKO JEŚLI ADMIN GO WŁĄCZYŁ)
# ==========================================
if st.session_state.show_debug:
    with st.sidebar:
        st.header("🔧 Tryb Testowy Badacza")
        if st.button("Wróć do Panelu Admina 🔐"):
            st.query_params["admin"] = "1"
            st.rerun()
        st.write(f"**ID Sesji:** `{st.session_state.session_id}`")
        st.write(f"**Język:** `{st.session_state.lang.upper()}` | **Krok:** {st.session_state.step} / 6")

        selected_mode = st.radio(
            "Wylosowany Tryb:",
            options=["B", "A"],
            format_func=lambda x: "Tryb B: Nemesis-LLM" if x == "B" else "Tryb A: Vanilla LLM",
            index=0 if st.session_state.mode == "B" else 1
        )
        st.session_state.mode = selected_mode

        st.divider()
        rel = st.session_state.relationship
        st.write(f"😡 Wrogość: {rel.wrogosc}/10 | 🤝 Wdzięczność: {rel.wdziecznosc}/10")
        st.write(f"🛡️ Szacunek: {rel.szacunek}/5 | ⚠️ Lęk: {rel.lek}/10")

        st.divider()
        st.write("**Ostatnio pobrane z RAG:**")
        if st.session_state.mode == "A":
            st.caption("Tryb A (RAG wyłączony)")
        elif st.session_state.last_rag_memories:
            for mem in st.session_state.last_rag_memories:
                st.caption(f"🧠 {mem}")
        else:
            st.caption("Brak wspomnień.")


# ==========================================
# WYBÓR JĘZYKA (TYLKO PL / EN) ORAZ PASEK POSTĘPU
# ==========================================
lang_col1, lang_col2 = st.columns([2, 1])
with lang_col2:
    lang_options = {"🇵🇱 Polski": "pl", "🇬🇧 English": "en"}
    current_idx = ["pl", "en"].index(st.session_state.lang)
    chosen_label = st.selectbox(
        "🌐 Język / Language:",
        options=list(lang_options.keys()),
        index=current_idx
    )
    new_lang = lang_options[chosen_label]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        if st.session_state.step == 4 and sum(1 for m in st.session_state.chat_history if m["role"] == "user") == 0:
            st.session_state.chat_history = []
        st.rerun()

L = TRANSLATIONS[st.session_state.lang]

step_weights = {
    1: 0.10,
    2: 0.25,
    3: 0.35,
    4: 0.40 + (st.session_state.episode * 0.12),
    5: 0.88,
    6: 1.00,
}
prog_val = step_weights.get(st.session_state.step, 0.10)
prog_text = L["progress_steps"][st.session_state.step].format(ep=st.session_state.episode)
st.progress(prog_val, text=prog_text)

if st.session_state.step in [2, 3, 4, 5]:
    inject_prevent_refresh_guard(enable=True)
    st.caption(L["no_refresh_banner"])
else:
    inject_prevent_refresh_guard(enable=False)


# ==========================================
# KROK 1: WPROWADZENIE I ŚWIADOMA ZGODA
# ==========================================
if st.session_state.step == 1:
    st.title(L["consent_title"])
    st.markdown(L["consent_body"])

    consent = st.checkbox(L["consent_check"])

    if st.button(L["consent_btn"], disabled=not consent, type="primary"):
        st.session_state.step = 2
        st.rerun()


# ==========================================
# KROK 2: PRETEST I METRYCZKA (SUWAKI 1–7 Z PODPISAMI NA DOLE)
# ==========================================
elif st.session_state.step == 2:
    st.title(L["pre_title"])
    st.write(L["pre_desc"])

    with st.form("pretest_form"):
        wiek = st.number_input(L["pre_age"], min_value=10, max_value=99, value=22, step=1)
        plec = st.radio(L["pre_gender"], L["pre_gender_opts"])
        czas_grania = st.radio(L["pre_hours"], L["pre_hours_opts"])
        gry_rpg = st.radio(L["pre_rpg"], L["pre_rpg_opts"], index=2)

        st.divider()
        st.markdown(L["pre_scale_info"])

        ai_1 = render_paper_likert(L["ai_1"], key="pre_ai_1", lang=st.session_state.lang, default_val=4)
        ai_2 = render_paper_likert(L["ai_2"], key="pre_ai_2", lang=st.session_state.lang, default_val=4)
        imm_1 = render_paper_likert(L["imm_1"], key="pre_imm_1", lang=st.session_state.lang, default_val=4)
        imm_2 = render_paper_likert(L["imm_2"], key="pre_imm_2", lang=st.session_state.lang, default_val=4)

        submitted = st.form_submit_button(L["pre_submit"], type="primary")

        if submitted:
            if wiek < 18 or wiek > 35:
                st.error(L["err_age"])
            elif gry_rpg == L["pre_rpg_opts"][0]:
                st.error(L["err_rpg"])
            else:
                st.session_state.pretest_data = {
                    "Wiek": wiek,
                    "Plec": plec,
                    "Czas_grania": czas_grania,
                    "Doswiadczenie_RPG": gry_rpg,
                    "AI_1": ai_1,
                    "AI_2": ai_2,
                    "Mean_AI_Attitude": round((ai_1 + ai_2) / 2, 2),
                    "IMM_1": imm_1,
                    "IMM_2": imm_2,
                    "Mean_Immersion_Trait": round((imm_1 + imm_2) / 2, 2),
                }
                st.session_state.step = 3
                st.rerun()


# ==========================================
# KROK 3: WPROWADZENIE FABULARNE
# ==========================================
elif st.session_state.step == 3:
    st.title(L["intro_title"])
    st.markdown(L["intro_body"])
    st.warning(L["no_refresh_banner"])

    if st.button(L["intro_btn"], type="primary"):
        st.session_state.step = 4
        st.rerun()


# ==========================================
# KROK 4: ROZGRYWKA WIELOEPIZODYCZNA (3 EPIZODY)
# ==========================================
elif st.session_state.step == 4:
    cur_ep = st.session_state.episode
    cur_lang = st.session_state.lang

    st.title(L["game_title"].format(ep=cur_ep))
    st.info(f"**{L['game_ctx_label']}** {EPISODE_CONTEXTS[cur_lang][cur_ep]}")

    if not st.session_state.chat_history:
        with st.spinner(L["spinner_approach"]):
            opening_memories = []
            if st.session_state.mode == "B" and cur_ep > 1:
                opening_memories = st.session_state.vector_memory.retrieve_relevant_memories(
                    query_text="dotychczasowa historia relacji obietnice zniewagi spotkanie history relationship",
                    n_results=4
                )
                st.session_state.last_rag_memories = opening_memories

            opening_text = generate_episode_opening(
                mode=st.session_state.mode,
                episode=cur_ep,
                relationship=st.session_state.relationship,
                retrieved_memories=opening_memories,
                lang=cur_lang
            )
            st.session_state.chat_history.append({"role": "assistant", "content": opening_text})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_turns = sum(1 for m in st.session_state.chat_history if m["role"] == "user")

    if player_input := st.chat_input(L["chat_placeholder"]):
        clean_input = player_input.strip()
        if len(clean_input.split()) < 2 or len(clean_input) < 5:
            st.warning(L["warn_short"])
        else:
            with st.chat_message("user"):
                st.markdown(clean_input)

            with st.spinner(L["spinner_reply"]):
                perception = analyze_player_input(clean_input, lang=cur_lang)
                st.session_state.perception_logs.append({
                    "ep": cur_ep,
                    "cat": perception.category,
                    "score": perception.affect_score,
                    "summary": perception.summary
                })

                retrieved_memories = []
                if st.session_state.mode == "B":
                    retrieved_memories = st.session_state.vector_memory.retrieve_relevant_memories(
                        query_text=clean_input,
                        n_results=4
                    )
                    st.session_state.last_rag_memories = retrieved_memories

                st.session_state.relationship.update(perception)

                npc_reply = generate_npc_response(
                    mode=st.session_state.mode,
                    episode=cur_ep,
                    chat_history=st.session_state.chat_history,
                    player_text=clean_input,
                    retrieved_memories=retrieved_memories,
                    relationship_state=st.session_state.relationship.to_Prompt_string(cur_lang),
                    lang=cur_lang
                )

                if st.session_state.mode == "B":
                    st.session_state.vector_memory.add_memory(
                        episode=cur_ep,
                        player_text=clean_input,
                        perception=perception,
                        lang=cur_lang
                    )

                rel = st.session_state.relationship
                st.session_state.all_chat_logs.append({
                    "session_id": st.session_state.session_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "lang": cur_lang,
                    "mode": st.session_state.mode,
                    "episode": cur_ep,
                    "turn_number": user_turns + 1,
                    "player_text": clean_input.replace(";", ",").replace("\n", " "),
                    "npc_reply": npc_reply.replace(";", ",").replace("\n", " "),
                    "category": perception.category,
                    "affect_score": perception.affect_score,
                    "summary": perception.summary.replace(";", ","),
                    "rag_retrieved": " | ".join(retrieved_memories).replace(";", ",") if retrieved_memories else "Brak",
                    "wrogosc_state": rel.wrogosc,
                    "wdziecznosc_state": rel.wdziecznosc,
                })

            st.session_state.chat_history.append({"role": "user", "content": clean_input})
            st.session_state.chat_history.append({"role": "assistant", "content": npc_reply})
            st.rerun()

    st.divider()
    if user_turns < 2:
        st.caption(L["turns_hint"].format(rem=2 - user_turns))
    else:
        if cur_ep < 3:
            if st.button(L["next_ep_btn"].format(ep=cur_ep, next_ep=cur_ep + 1), type="primary"):
                st.session_state.episode += 1
                st.session_state.chat_history = []
                st.rerun()
        else:
            if st.button(L["end_game_btn"], type="primary"):
                st.session_state.step = 5
                st.rerun()


# ==========================================
# KROK 5: POSTTEST (SUWAKI 1–7 ZE STAŁYMI PODPISAMI POD KAŻDĄ SKALĄ)
# ==========================================
elif st.session_state.step == 5:
    cur_lang = st.session_state.lang
    st.title(L["post_title"])
    st.write(L["post_desc"])

    with st.form("posttest_form"):
        st.subheader(L["post_sec2"])

        ag_1 = render_paper_likert(L["ag_1"], key="ag_1", lang=cur_lang, default_val=4)
        ag_2 = render_paper_likert(L["ag_2"], key="ag_2", lang=cur_lang, default_val=4)
        ag_3 = render_paper_likert(L["ag_3"], key="ag_3", lang=cur_lang, default_val=4)
        ag_4 = render_paper_likert(L["ag_4"], key="ag_4", lang=cur_lang, default_val=4)
        ag_5 = render_paper_likert(L["ag_5"], key="ag_5", lang=cur_lang, default_val=4)

        rel_1 = render_paper_likert(L["rel_1"], key="rel_1", lang=cur_lang, default_val=4)
        rel_2 = render_paper_likert(L["rel_2"], key="rel_2", lang=cur_lang, default_val=4)
        rel_3 = render_paper_likert(L["rel_3"], key="rel_3", lang=cur_lang, default_val=4)
        rel_4 = render_paper_likert(L["rel_4"], key="rel_4", lang=cur_lang, default_val=4)

        st.divider()
        st.subheader(L["post_sec3"])
        st.caption(L["post_sec3_cap"])

        god_1 = render_paper_semantic(L["god_1_title"], L["god_1_left"], L["god_1_right"], key="god_1", lang=cur_lang, default_val=4)
        god_2 = render_paper_semantic(L["god_2_title"], L["god_2_left"], L["god_2_right"], key="god_2", lang=cur_lang, default_val=4)
        god_3 = render_paper_semantic(L["god_3_title"], L["god_3_left"], L["god_3_right"], key="god_3", lang=cur_lang, default_val=4)
        god_4 = render_paper_semantic(L["god_4_title"], L["god_4_left"], L["god_4_right"], key="god_4", lang=cur_lang, default_val=4)

        casa_1 = render_paper_likert(L["casa_1"], key="casa_1", lang=cur_lang, default_val=4)
        casa_2 = render_paper_likert(L["casa_2"], key="casa_2", lang=cur_lang, default_val=4)
        casa_3 = render_paper_likert(L["casa_3"], key="casa_3", lang=cur_lang, default_val=4)

        st.divider()
        st.subheader(L["post_sec4"])

        tom_1 = render_paper_likert(L["tom_1"], key="tom_1", lang=cur_lang, default_val=4)
        tom_2 = render_paper_likert(L["tom_2"], key="tom_2", lang=cur_lang, default_val=4)
        coh_1 = render_paper_likert(L["coh_1"], key="coh_1", lang=cur_lang, default_val=4)
        coh_2 = render_paper_likert(L["coh_2"], key="coh_2", lang=cur_lang, default_val=4)
        hal_1 = render_paper_likert(L["hal_1"], key="hal_1", lang=cur_lang, default_val=2)

        hal_options = L["hal_2_opts"]
        st.markdown(f'<div class="paper-question-title">{L["hal_2_q"]}</div>', unsafe_allow_html=True)
        hal_2 = st.radio(L["hal_2_q"], hal_options, label_visibility="collapsed")

        st.markdown(f'<div class="paper-question-title">{L["open_q"]}</div>', unsafe_allow_html=True)
        open_q = st.text_area(L["open_q"], label_visibility="collapsed")

        finish_btn = st.form_submit_button(L["post_submit"], type="primary")

        if finish_btn:
            posttest_results = {
                "AG_1": ag_1, "AG_2": ag_2, "AG_3": ag_3, "AG_4": ag_4, "AG_5": ag_5,
                "Mean_Agency": round((ag_1 + ag_2 + ag_3 + ag_4 + ag_5) / 5, 2),
                "REL_1": rel_1, "REL_2": rel_2, "REL_3": rel_3, "REL_4": rel_4,
                "Mean_Relatedness": round((rel_1 + rel_2 + rel_3 + rel_4) / 4, 2),
                "GOD_1": god_1, "GOD_2": god_2, "GOD_3": god_3, "GOD_4": god_4,
                "Mean_Godspeed_Anthropomorphism": round((god_1 + god_2 + god_3 + god_4) / 4, 2),
                "CASA_1": casa_1, "CASA_2": casa_2, "CASA_3": casa_3,
                "Mean_CASA_SocialPresence": round((casa_1 + casa_2 + casa_3) / 3, 2),
                "ToM_1": tom_1, "ToM_2": tom_2,
                "Mean_ToM": round((tom_1 + tom_2) / 2, 2),
                "COH_1": coh_1, "COH_2": coh_2,
                "Mean_Coherence": round((coh_1 + coh_2) / 2, 2),
                "HAL_1": hal_1,
                "HAL_2_Kat": hal_2,
                "HAL_2_Kod": hal_options.index(hal_2),
                "Uwagi_Otwarte": open_q.replace(";", ",").replace("\n", " ")
            }
            if not st.session_state.data_saved:
                save_study_results(posttest_results)
                st.session_state.data_saved = True
            st.session_state.step = 6
            st.rerun()


# ==========================================
# KROK 6: DEBRIEFING I ZAKOŃCZENIE
# ==========================================
elif st.session_state.step == 6:
    st.success(L["debrief_saved"])
    st.title(L["debrief_title"])
    st.markdown(L["debrief_body"])