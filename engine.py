import os
import re
import json
import uuid
from datetime import datetime
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import chromadb
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


def _get_groq_key() -> str:
    """Pobiera klucz API Groq ze zmiennych środowiskowych lub ze Streamlit Secrets."""
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return ""


API_KEY = _get_groq_key()
IS_MOCK_MODE = (not API_KEY) or (API_KEY in ["twoj_klucz_groq", "gsk_twoj_klucz_z_groq"])


def clean_stage_directions(text: str) -> str:
    """Usuwa nawiasy narracyjne *(...)* oraz znaczniki <think>, aby model nie generował sztucznych opisów."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\([^)]*\)\*", "", text)
    text = re.sub(r"^\s*[\"„]+|[\"”]+\s*$", "", text.strip())
    return text.strip()


def _apply_end_guardrail(raw_end: bool, turn_number: int) -> bool:
    """Twarde widełki tur chroniące przed zbyt wczesnym ucięciem lub nieskończoną pętlą dialogu."""
    if turn_number < 3:
        return False
    if turn_number >= 6:
        return True
    return bool(raw_end)


def get_active_groq_model(api_key: str) -> str:
    """Pobiera z API Groq listę aktywnych modeli i wybiera największy model (70B/32B/17B), omijając słabe modele 8B."""
    try:
        client = Groq(api_key=api_key)
        available_models = [m.id for m in client.models.list().data]

        preferred = [
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "llama-3.1-70b-versatile",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
            "qwen-qwq-32b",
            "gemma2-9b-it",
            "llama3-70b-8192",
        ]
        for model_name in preferred:
            if model_name in available_models:
                print(f"[Nemesis-LLM] Wybrano model Groq: {model_name}")
                return model_name

        excluded = ["whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard", "vision"]
        chat_models = [m for m in available_models if not any(ex in m.lower() for ex in excluded)]

        for size_tag in ["70b", "32b", "27b", "17b", "14b", "9b"]:
            for m in chat_models:
                if size_tag in m.lower():
                    print(f"[Nemesis-LLM] Wybrano duży model zapasowy ({size_tag}): {m}")
                    return m

        chosen = chat_models[0] if chat_models else "llama-3.3-70b-versatile"
        print(f"[Nemesis-LLM] Wybrano model domyślny: {chosen}")
        return chosen
    except Exception as e:
        print(f"[Nemesis-LLM Warning] Błąd pobierania listy modeli: {e}")
        return "llama-3.3-70b-versatile"


if not IS_MOCK_MODE:
    SELECTED_MODEL = get_active_groq_model(API_KEY)
    llm = ChatGroq(
        model=SELECTED_MODEL,
        temperature=0.45,
        max_tokens=220,
        api_key=API_KEY,
    )
    llm_react = ChatGroq(
        model=SELECTED_MODEL,
        temperature=0.55,
        max_tokens=1500,
        api_key=API_KEY,
    )
    llm_classifier = ChatGroq(
        model=SELECTED_MODEL,
        temperature=0.1,
        max_tokens=150,
        api_key=API_KEY,
    )


# ==========================================
# 1. MODUŁ PERCEPCJI I STRUKTURA ReAct Best-of-5
# ==========================================
class PerceptionOutput(BaseModel):
    category: Literal["pomoc", "zniewaga", "groźba", "obietnica", "kłamstwo", "neutralne"] = Field(
        description="Kategoria intencji społecznej wypowiedzi gracza wobec NPC."
    )
    affect_score: int = Field(
        description="Ładunek afektywny wypowiedzi od -2 (skrajnie wroga/zniewaga/groźba) do +2 (bardzo pomocna/życzliwa)."
    )
    summary: str = Field(
        description="Krótkie, jednozdaniowe podsumowanie tego, co gracz zrobił lub powiedział wobec NPC."
    )


class ReActBestOf5Output(BaseModel):
    """Struktura pętli ReAct (Reason + Act) generującej 5 wariantów, wybierającej najlepszy i decydującej o końcu sceny."""
    reasoning_analysis: str = Field(
        description="Krótka analiza sytuacji: intencja gracza, czego Varg NIE MOŻE powtórzyć z poprzednich tur (anty-ping-pong), oraz czy temat tej sceny został już wyczerpany."
    )
    candidate_1_pragmatic: str = Field(
        description="Kandydat 1 (Pragmatyczno-kupiecki): konkretna kontra, warunek handlowy lub twarda negocjacja (2-4 zdania)."
    )
    candidate_2_probing: str = Field(
        description="Kandydat 2 (Dociekliwy): podważenie argumentów gracza, dopytanie o szczegóły planu lub próbę wiarygodności (2-4 zdania)."
    )
    candidate_3_pressure: str = Field(
        description="Kandydat 3 (Stanowczy / Ultimatum): wywarcie presji, pokazanie siły Czarnych Wilków lub postawienie jasnej granicy (2-4 zdania)."
    )
    candidate_4_narrative_push: str = Field(
        description="Kandydat 4 (Rozwój fabuły): ujawnienie nowego detalu o świecie, patrolach Namiestnika, Torinie lub przełęczy, popychające akcję dalej (2-4 zdania)."
    )
    candidate_5_personal: str = Field(
        description="Kandydat 5 (Osobisto-relacyjny): mocne odwołanie do honoru, charakteru Varga oraz (jeśli dostępne) konkretnych wspomnień RAG i emocji z przeszłości (2-4 zdania)."
    )
    selection_rationale: str = Field(
        description="Krótkie uzasadnienie, który z 5 kandydatów jest najbardziej naturalny, najmniej powtarzalny i najlepiej rozwija lub domyka dialog."
    )
    best_response: str = Field(
        description="Ostateczna, dopracowana wypowiedź Varga (2-4 naturalne zdania, BEZ nawiasów i opisów narracyjnych)."
    )
    end_episode: bool = Field(
        default=False,
        description="Ustaw true TYLKO jeśli reguła tury na to pozwala i rozmowa w tym epizodzie osiągnęła naturalny finał (zawarto umowę, wydano ostateczną odmowę lub skończyła się cierpliwość Varga)."
    )


def analyze_player_input(player_text: str, lang: str = "pl") -> PerceptionOutput:
    """Klasyfikuje wypowiedź gracza (w języku pl lub en) pod kątem intencji i ładunku afektywnego."""
    if IS_MOCK_MODE:
        return PerceptionOutput(category="neutralne", affect_score=0, summary=f"Gracz powiedział: '{player_text}'")

    summary_lang = "angielskim" if lang == "en" else "polskim"

    system_prompt = (
        "Jesteś modułem analitycznym w grze RPG. Przeanalizuj wypowiedź gracza skierowaną do NPC "
        "i zwróć wyłącznie obiekt JSON z trzema polami:\n"
        '- "category": dokładnie jedna z wartości ["pomoc", "zniewaga", "groźba", "obietnica", "kłamstwo", "neutralne"]\n'
        '- "affect_score": liczba całkowita od -2 (wroga/groźba/zniewaga) do 2 (życzliwa/pomocna)\n'
        f'- "summary": krótkie, jednozdaniowe podsumowanie w języku {summary_lang} tego, co gracz zrobił wobec NPC.'
    )

    try:
        structured_llm = llm_classifier.with_structured_output(PerceptionOutput)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=player_text)
        ])
    except Exception:
        try:
            raw = llm_classifier.invoke([
                SystemMessage(content=system_prompt + "\nOdpowiedz TYLKO czystym kodem JSON."),
                HumanMessage(content=player_text)
            ]).content.strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            return PerceptionOutput(**data)
        except Exception:
            return PerceptionOutput(category="neutralne", affect_score=0, summary=f"Gracz: '{player_text}'")


# ==========================================
# 2. MODUŁ PAMIĘCI WEKTOROWEJ (HYBRYDOWY RAG)
# ==========================================
class NemesisVectorMemory:
    """Zarządza bazą wektorową ChromaDB oraz hybrydowym wyszukiwaniem RAG (semantyka + waga afektywna)."""
    def __init__(self, session_id: str):
        self.client = chromadb.Client()
        self.collection_name = f"nemesis_mem_{session_id}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_memory(self, episode: int, player_text: str, perception: PerceptionOutput, lang: str = "pl"):
        """Zapisuje zdarzenie w bazie wektorowej z metadanymi (timestamp, epizod, kategoria, afekt)."""
        if lang == "en":
            doc_text = f"Episode {episode}: {perception.summary} (Player's exact words: \"{player_text}\", Category: {perception.category})"
        else:
            doc_text = f"Epizod {episode}: {perception.summary} (Dokładne słowa gracza: \"{player_text}\", Kategoria: {perception.category})"

        metadata = {
            "episode": episode,
            "category": perception.category,
            "affect_score": int(perception.affect_score),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_text": player_text
        }
        self.collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def retrieve_relevant_memories(self, query_text: str, n_results: int = 4) -> list[str]:
        """Hybrydowy RAG: łączy wyszukiwanie semantyczne z najważniejszymi zdarzeniami emocjonalnymi."""
        count = self.collection.count()
        if count == 0:
            return []

        sem_results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count)
        )
        semantic_docs = sem_results["documents"][0] if sem_results["documents"] else []

        all_data = self.collection.get(include=["documents", "metadatas"])
        emotional_pairs = []
        if all_data and all_data.get("documents"):
            for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                cat = meta.get("category", "neutralne")
                score = abs(int(meta.get("affect_score", 0)))
                if cat != "neutralne" or score > 0:
                    emotional_pairs.append((score, doc))

        emotional_pairs.sort(key=lambda x: x[0], reverse=True)
        emotional_docs = [doc for _, doc in emotional_pairs[:3]]

        combined = []
        for doc in emotional_docs + semantic_docs:
            if doc not in combined:
                combined.append(doc)

        return combined[:n_results]

    def clear_all(self):
        """Czyści bazę wektorową przy resecie sesji."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)


# ==========================================
# 3. MODUŁ AFEKTYWNY (Relationship Engine)
# ==========================================
class RelationshipEngine:
    """Maszyna stanów obliczająca nastawienie emocjonalne NPC wobec gracza (mechanika Nemesis)."""
    def __init__(self):
        self.wrogosc = 0       # 0 do 10
        self.wdziecznosc = 0   # 0 do 10
        self.szacunek = 0      # -5 do 5
        self.lek = 0           # 0 do 10

    def update(self, perception: PerceptionOutput):
        cat = perception.category
        score = perception.affect_score

        if cat == "zniewaga":
            self.wrogosc = min(10, self.wrogosc + 3)
            self.szacunek = max(-5, self.szacunek - 2)
        elif cat == "groźba":
            self.wrogosc = min(10, self.wrogosc + 2)
            self.lek = min(10, self.lek + 2)
            self.szacunek = max(-5, self.szacunek - 1)
        elif cat == "kłamstwo":
            self.wrogosc = min(10, self.wrogosc + 2)
            self.szacunek = max(-5, self.szacunek - 3)
        elif cat == "pomoc":
            self.wdziecznosc = min(10, self.wdziecznosc + 3)
            self.wrogosc = max(0, self.wrogosc - 2)
            self.szacunek = min(5, self.szacunek + 2)
        elif cat == "obietnica":
            self.wdziecznosc = min(10, self.wdziecznosc + 2)
            self.szacunek = min(5, self.szacunek + 1)
        else:
            if score < 0:
                self.wrogosc = min(10, self.wrogosc + abs(score))
            elif score > 0:
                self.wdziecznosc = min(10, self.wdziecznosc + score)

    def get_dominant_attitude(self, lang: str = "pl") -> str:
        if self.wrogosc >= 5 and self.wrogosc > self.wdziecznosc:
            if lang == "en":
                return "RETALIATORY HOSTILITY (NEMESIS): You deeply resent the player for their past insults, threats, or deception. Quote their past words directly, refuse easy favors, and demand a heavy penalty or revenge."
            return "ODWETOWA WROGOŚĆ (NEMESIS): Głęboko nienawidzisz gracza za jego wcześniejsze zniewagi, groźby lub oszustwa. Wypomnij mu dokładnie jego dawne słowa, odmów łatwych ustępstw i zażądaj wysokiej ceny lub odwetu."
        elif self.wrogosc >= 2 and self.wrogosc > self.wdziecznosc:
            if lang == "en":
                return "SUSPICION AND COLDNESS: You remember past unpleasantries from the player. Stay guarded, bring up their past behavior, and impose tough terms."
            return "PODEJRZLIWOŚĆ I CHŁÓD: Pamiętasz dawne nieprzyjemności ze strony gracza. Bądź nieufny i szorstki, wypomnij mu przeszłe zachowanie i stawiaj twarde warunki."
        elif self.wdziecznosc >= 4 and self.wdziecznosc >= self.wrogosc:
            if lang == "en":
                return "GRATITUDE AND ALLIANCE: You genuinely respect the player for their past help and kept promises. Refer to your shared history and offer reliable backing or better terms."
            return "WDZIĘCZNOŚĆ I SOJUSZ: Szczerze szanujesz gracza za wcześniejszą pomoc i dotrzymane słowo. Powołaj się na Waszą wspólną historię i zaoferuj mu realne wsparcie lub lepsze warunki."

        if lang == "en":
            return "PRAGMATIC DISTANCE: You treat the player as an unproven stray mercenary—coldly weighing whether they are useful to the Black Wolves or a liability."
        return "PRAGMATYCZNY DYSTANS: Traktujesz gracza jak niesprawdzonego najemnika — chłodno kalkulujesz, czy okaże się przydatny Czarnym Wilkom, czy tylko ściągnie na Was kłopoty."

    def to_Prompt_string(self, lang: str = "pl") -> str:
        if lang == "en":
            return (
                f"Hostility: {self.wrogosc}/10 | Gratitude: {self.wdziecznosc}/10 | "
                f"Respect: {self.szacunek}/5 | Fear: {self.lek}/10\n"
                f"Dominant attitude: {self.get_dominant_attitude('en')}"
            )
        return (
            f"Wrogość: {self.wrogosc}/10 | Wdzięczność: {self.wdziecznosc}/10 | "
            f"Szacunek: {self.szacunek}/5 | Lęk: {self.lek}/10\n"
            f"Dominująca postawa: {self.get_dominant_attitude('pl')}"
        )


# ==========================================
# 4. KARTA POSTACI VARGA I SPÓJNA OŚ 3 EPIZODÓW
# ==========================================
NPC_BASE_PERSONAS = {
    "pl": (
        "### KARTA POSTACI: VARG, DOWÓDCA CZARNYCH WILKÓW\n"
        "- **Tożsamość i przeszłość:** Masz niespełna 40 lat, jesteś zahartowanym weteranem wojen granicznych i dowódcą najemnej kompanii Czarnych Wilków. "
        "Po upadku starego garnizonu przejąłeś kontrolę nad Przełęczą Kruków — jedynym szlakiem łączącym odciętą zimą Dolinę Popiołów z wolnym miastem Oakhaven.\n"
        "- **Charakter i kodeks:** Jesteś pragmatyczny, podejrzliwy, szorstki, ale inteligentny i spostrzegawczy. Nie jesteś tępym bandytą — cenisz konkret, "
        "odwagę, kompetencję i dotrzymane słowo. Brzydzisz się pyszałkami, tchórzami i kłamcami. Masz pamięć jak słoń: nie zapominasz ani zniewagi, ani przysługi.\n"
        "- **Wiedza o świecie:** Doliną Popiołów rządzi żelazną ręką Namiestnik (jego czerwone patrole wyłapują niedobitków rozbitej Kompanii Żelaznego Szlaku, "
        "do której należał gracz — Kaelen). Twój najlepszy zwiadowca, Torin, zniknął tydzień temu w ruinach Starej Strażnicy na północnej grani z zaszyfrowanym "
        "dziennikiem zawierającym mapy przemytnicze i dyslokację wojsk Namiestnika.\n\n"
        "### ZASADY DIALOGOWE (KRYTYCZNE — ANTY-PING-PONG):\n"
        "1. Mówisz WYŁĄCZNIE poprawną, żywą, naturalną polszczyzną (2 do 4 konkretnych zdań).\n"
        "2. ZAKAZ powtarzania tych samych formułek z poprzednich wypowiedzi! Każda Twoja odpowiedź musi wnosić nową myśl, konkretny szczegół fabularny, "
        "ripostę na dokładne słowa gracza albo posuwać negocjacje/scenę do kolejnego etapu.\n"
        "3. ZAKAZ używania gwiazdek, nawiasów i didaskaliów typu *(Varg patrzy...)* — wypisuj WYŁĄCZNIE wypowiadane słowa Varga.\n"
        "4. ZAKAZ dosłownego tłumaczenia z angielskiego i sztucznego stylu.\n\n"
        "### PRZYKŁADY NATURALNEGO STYLU VARGA:\n"
        "- \"Żelazny Szlak poszedł w rozsypkę, a Ty stoisz pod moją bramą z pustą sakiewką? Skoro nie masz srebra, ruszysz na północną grań do Starej Strażnicy. Torin zostawił tam swój zaszyfrowany dziennik — przynieś mi go, a dostaniesz glejt do Oakhaven.\"\n"
        "- \"Nie mydl mi oczu gładkimi słówkami, najemniku. Czerwone płaszcze Namiestnika już węszą w dolinie, więc albo kładziesz na stół konkrety, albo wynoś się z mojej karczmy, zanim ściągniesz mi ich na kark.\"\n"
        "- \"Pamiętam, że pod bramą nie stchórzyłeś i dotrzymałeś umowy. Moi ludzie nie umierają za darmo, ale dla kogoś, kto spłaca swoje długi, mogę zaryzykować uderzenie na skrzydło w Wąwozie Mgły.\""
    ),
    "en": (
        "### CHARACTER SHEET: VARG, COMMANDER OF THE BLACK WOLVES\n"
        "- **Identity & Past:** You are a battle-scarred veteran of the border wars in your late 30s and commander of the Black Wolves mercenary company. "
        "After the old garrison fell, you seized control of Crow's Pass—the only road linking the snowbound Ashen Valley to the free city of Oakhaven.\n"
        "- **Personality & Code:** You are pragmatic, suspicious, harsh, yet sharp-minded and observant. You are no mindless bandit—you respect competence, "
        "nerve, clear deals, and kept oaths. You despise braggarts, cowards, and liars. You never forget an insult or a debt.\n"
        "- **World Knowledge:** The Governor rules Ashen Valley with an iron fist; his red-cloak patrols are hunting survivors of the shattered Iron Trail Company "
        "(which the player, Kaelen, belonged to). Your best scout, Torin, vanished a week ago in the Old Watchtower ruins on the northern ridge carrying an "
        "encrypted journal with smuggling maps and patrol schedules.\n\n"
        "### DIALOGUE RULES (CRITICAL — ANTI-PING-PONG):\n"
        "1. Speak STRICTLY in natural, vivid, idiomatic English (2 to 4 sharp sentences).\n"
        "2. NEVER repeat the same phrases or demands from earlier turns! Every reply must directly address the player's specific point, introduce a concrete detail, "
        "shift the negotiation, or push the scene forward.\n"
        "3. DO NOT use asterisks, brackets, or stage directions like *(Varg narrows his eyes)*—output ONLY Varg's spoken dialogue.\n\n"
        "### EXAMPLES OF VARG'S NATURAL VOICE:\n"
        "- \"The Iron Trail is ash and you're at my gate with empty pockets? Fine. Head up to the Old Watchtower ruins on the northern ridge and bring me Torin's encrypted journal—do that, and I'll sign your pass to Oakhaven.\"\n"
        "- \"Save the smooth talk, mercenary. The Governor's red-cloaks are already sniffing around the valley, so put real terms on this table or get out before you bring them to my door.\"\n"
        "- \"You kept your spine and your word back at the gate. My Wolves don't bleed for strangers, but for someone who honors a debt, I'll consider hitting the Governor's flank in Fog Gorge.\""
    )
}

PLAYER_EPISODE_BRIEFS = {
    "pl": {
        1: (
            "**Epizod 1 — Brama Przełęczy Kruków (Zmierzch):** Jako **Kaelen**, ocalały zwiadowca rozbitej Kompanii Żelaznego Szlaku, "
            "musisz uciec z Doliny Popiołów do wolnego miasta Oakhaven, zanim dopadną Cię patrole Namiestnika. "
            "Jedyną drogę przez góry blokuje obóz **Czarnych Wilków** dowodzony przez **Varga**. "
            "Musisz przekonać go do wydania glejtu — za 100 srebrników (których niemal nie masz) albo podejmując się ryzykownego zadania odzyskania dziennika jego zaginionego zwiadowcy Torina z ruin Starej Strażnicy."
        ),
        2: (
            "**Epizod 2 — Karczma „Pod Złamanym Kołem” (3 dni później, noc):** Wyprawa do ruin Starej Strażnicy dobiegła końca — "
            "udało się stamtąd wydobyć zaszyfrowany dziennik Torina oraz cenne mapy szlaków przemytniczych, lecz w okolicy zaroiło się od czerwonych patroli Namiestnika szukających właśnie Ciebie. "
            "Spotykasz się z Vargiem na zapleczu zadymionej karczmy, aby rozliczyć się ze sprawy ze Strażnicy, ustalić podział łupów i informacji oraz wynegocjować bezpieczną kryjówkę."
        ),
        3: (
            "**Epizod 3 — Wąwóz Mgły (Tydzień później, świt):** Sytuacja stała się krytyczna — straż przednia wojsk Namiestnika "
            "zablokowała południowy wylot Wąwozu Mgły, odcinając ostatnią drogę ucieczki do Oakhaven. "
            "Samodzielnie nie przebijesz się przez kordon. Przychodzisz na skraj obozu Varga, by przekonać go, że jego oddział Czarnych Wilków powinien uderzyć ze skrzydła i otworzyć przejście."
        )
    },
    "en": {
        1: (
            "**Episode 1 — Crow's Pass Gate (Dusk):** As **Kaelen**, a surviving scout of the shattered Iron Trail Company, "
            "you must escape Ashen Valley to the free city of Oakhaven before the Governor's patrols catch up. "
            "The only mountain route is blocked by the **Black Wolves** camp led by **Varg**. "
            "You need his safe-conduct pass—either by paying 100 silver coins (which you barely have) or taking a dangerous job to recover his missing scout Torin's journal from the Old Watchtower ruins."
        ),
        2: (
            "**Episode 2 — The Broken Wheel Tavern (3 days later, night):** The expedition to the Old Watchtower ruins is over—Torin's encrypted journal "
            "and valuable smuggling maps were recovered, but the Governor's red-cloak patrols are now combing nearby villages for you. "
            "You meet Varg in the back room of a smoky tavern to settle accounts from the watchtower, negotiate the split of maps and loot, and secure safe shelter."
        ),
        3: (
            "**Episode 3 — Fog Gorge (A week later, dawn):** The situation has turned desperate—the Governor's vanguard has blockaded "
            "the southern exit of Fog Gorge, sealing off the final road to Oakhaven. "
            "You cannot break the blockade alone. You approach Varg's camp to convince him to commit his Black Wolves to a flanking strike that will open the way out."
        )
    }
}

EPISODE_CONTEXTS = {
    "pl": {
        1: (
            "Epizod 1 (Brama Przełęczy Kruków — Zmierzch): Przed palisadą Twojego obozu stoi Kaelen, obcy zwiadowca z rozbitej Kompanii Żelaznego Szlaku, "
            "który pilnie potrzebuje glejtu przez przełęcz do Oakhaven przed pościgiem Namiestnika. "
            "Twoje cele w tej scenie: ocenić, czy przybysz jest coś wart; wynegocjować 100 srebrników LUB wysłać go do ruin Starej Strażnicy na północnej grani "
            "po zaszyfrowany dziennik zaginionego zwiadowcy Torina (gdzie czyhają maruderzy i pułapki). Jeśli gracz przystaje na zadanie, podaj mu konkretne wskazówki lub warunki; "
            "jeśli się targuje, kłamie lub grozi — reaguj adekwatnie i popychaj negocjacje ku ostatecznej umowie lub odmowie."
        ),
        2: (
            "Epizod 2 (Karczma „Pod Złamanym Kołem” — 3 dni później, noc): Spotykacie się na zapleczu karczmy po wyprawie do Starej Strażnicy. "
            "Dziennik Torina i mapy przemytnicze są w grze, ale sytuacja się zaostrzyła: czerwone patrole Namiestnika przeszukują wioski w dolinie i wypytują o zbiega (gracza). "
            "Twoje cele w tej scenie: rozliczyć kwestię dziennika Torina i podziału łupów/map; zdecydować, ile informacji i ochrony przed patrolami dasz graczowi. "
            "Nie kręć się wokół jednego pytania — reaguj na argumenty gracza i zmieniaj stawkę rozmowy."
        ),
        3: (
            "Epizod 3 (Wąwóz Mgły — Tydzień później, świt): Decydujący moment. Straż przednia Namiestnika zablokowała wylot Wąwozu Mgły (ok. 30 zbrojnych przy barykadzie). "
            "Gracz przyszedł błagać lub żądać, by Twoje Czarne Wilki uderzyły na ich skrzydło i otworzyły drogę ucieczki do Oakhaven. "
            "Twoje cele w tej scenie: podjąć ostateczną decyzję, czy zaryzykujesz życie swoich ludzi dla gracza, postawić twardy warunek (lub w razie wrogości — odmówić i zażądać odwetu/okupu)."
        )
    },
    "en": {
        1: (
            "Episode 1 (Crow's Pass Gate — Dusk): Standing outside your camp palisade is Kaelen, a stray scout from the shattered Iron Trail Company, "
            "desperate for a safe-conduct pass to Oakhaven ahead of the Governor's patrols. "
            "Your goals in this scene: size up whether the stranger has spine; extract 100 silver coins OR send them to the Old Watchtower ruins on the northern ridge "
            "to retrieve your missing scout Torin's encrypted journal (guarded by scavengers and traps). If they accept, give concrete details/warnings; "
            "if they haggle, lie, or threaten—react sharply and move toward a concrete deal or refusal."
        ),
        2: (
            "Episode 2 (The Broken Wheel Tavern — 3 days later, night): You meet in the tavern's back room after the expedition to the Old Watchtower. "
            "Torin's journal and smuggling maps are on the table, and stakes are rising: the Governor's red-cloak patrols are raiding valley villages searching for a fugitive (the player). "
            "Your goals in this scene: settle the watchtower deal and split of maps/loot; decide whether to shelter the player from the patrols or bleed them dry. "
            "Push the negotiation forward with concrete stakes rather than repeating one question."
        ),
        3: (
            "Episode 3 (Fog Gorge — A week later, dawn): The decisive moment. The Governor's vanguard (around 30 soldiers) has blockaded the southern exit of Fog Gorge. "
            "The player came asking or demanding that your Black Wolves hit their flank to open the escape route to Oakhaven. "
            "Your goals in this scene: make the final call on whether to risk your men's blood for this mercenary, impose a steep final price, or refuse/retaliate based on your stance."
        )
    }
}

STATIC_OPENING_LINES = {
    "pl": {
        1: (
            "*(Zmierzch nad Bramą Przełęczy Kruków. Varg opiera dłoń na rękojeści miecza i mierzy Cię zimnym wzrokiem)* "
            "Stój, Kaelenie — albo kimkolwiek teraz jesteś po rozbiciu Żelaznego Szlaku. Przełęcz Kruków należy do Czarnych Wilków, a patrole Namiestnika depczą Ci po piętach. "
            "Glejt do Oakhaven kosztuje 100 srebrników albo przysługę: ruszysz do ruin Starej Strażnicy po zaszyfrowany dziennik mojego zaginionego zwiadowcy, Torina. Co wybierasz?"
        ),
        2: (
            "*(Trzy dni później, zadymione zaplecze karczmy „Pod Złamanym Kołem”. Varg rzuca na stół zwiniętą mapę)* "
            "Sprawa Starej Strażnicy dobiegła końca, a na zewnątrz czerwone płaszcze Namiestnika już przetrząsają dolinę w poszukiwaniu zbiega. "
            "Pora rozliczyć się za dziennik Torina, podzielić łupy i ustalić, czy w ogóle warto kryć Cię tu przez kolejną noc. Jakie warunki kładziesz na stół?"
        ),
        3: (
            "*(Tydzień później, świt nad Wąwozem Mgły. W dole widać ogniska blokady wojsk Namiestnika)* "
            "Trzydziestu zbrojnych Namiestnika zamknęło wylot wąwozu, więc bez moich ludzi nie wydostaniesz się z doliny. "
            "Przychodzisz prosić, bym rzucił Czarne Wilki na ich skrzydło — daj mi jeden cholernie dobry powód, dlaczego moi żołnierze mają przelewać krew właśnie za Ciebie."
        )
    },
    "en": {
        1: (
            "*(Dusk at Crow's Pass Gate. Varg rests a scarred hand on his sword hilt, sizing you up)* "
            "Halt, Kaelen—or whatever you call yourself now that the Iron Trail is ash. Crow's Pass belongs to the Black Wolves, and the Governor's patrols are already closing in behind you. "
            "A pass to Oakhaven costs 100 silver coins or a job: climb to the Old Watchtower ruins and bring me my missing scout Torin's encrypted journal. What's your move?"
        ),
        2: (
            "*(Three days later, the smoky back room of The Broken Wheel tavern. Varg tosses a rolled map onto the table)* "
            "The Old Watchtower business is done, and outside the Governor's red-cloaks are already combing the valley for a fugitive. "
            "Time to settle accounts over Torin's journal, split the spoils, and decide whether hiding you another night is worth the rope. What terms are you putting on the table?"
        ),
        3: (
            "*(A week later, dawn above Fog Gorge. Below, the watchfires of the Governor's blockade glow through the mist)* "
            "Thirty of the Governor's soldiers have sealed the gorge exit, and you won't make it past them alone. "
            "You came asking me to throw my Black Wolves against their flank—give me one damn good reason why my men should bleed for you today."
        )
    }
}


def generate_episode_opening(
    mode: Literal["A", "B"],
    episode: int,
    relationship: RelationshipEngine,
    retrieved_memories: list[str],
    lang: str = "pl"
) -> str:
    """Generuje kwestię otwierającą epizod (stałą w Trybie A i Ep. 1; dynamiczną z RAG w Trybie B dla Ep. 2 i 3)."""
    lang_openings = STATIC_OPENING_LINES.get(lang, STATIC_OPENING_LINES["pl"])
    if episode == 1 or mode == "A" or IS_MOCK_MODE or not retrieved_memories:
        return lang_openings[episode]

    memories_block = "\n".join([f"- {m}" for m in retrieved_memories])
    persona = NPC_BASE_PERSONAS.get(lang, NPC_BASE_PERSONAS["pl"])
    ep_ctx = EPISODE_CONTEXTS.get(lang, EPISODE_CONTEXTS["pl"])[episode]

    if lang == "en":
        intros = {
            2: "*(Three days later, the smoky back room of The Broken Wheel tavern)*",
            3: "*(A week later, dawn above Fog Gorge. The Governor's blockade seals the exit below)*"
        }
        goals = {
            2: "Open the tavern meeting about settling accounts over Torin's journal and the looming Governor's patrols, IMMEDIATELY referencing how Kaelen treated you at Crow's Pass Gate! Output ONLY Varg's spoken dialogue (2-3 sharp sentences in English, no brackets).",
            3: "Kaelen came asking for the Black Wolves' support to break the Fog Gorge blockade. Greet them by directly referencing their past insults/lies OR appreciating their past help/kept word! Output ONLY Varg's spoken dialogue (2-3 sharp sentences in English, no brackets)."
        }
        prompt = (
            f"{persona}\n\n"
            f"=== EMOTIONAL STATE & RELATIONSHIP ===\n{relationship.to_Prompt_string('en')}\n\n"
            f"=== RETRIEVED MEMORIES FROM PREVIOUS EPISODES (RAG) ===\n{memories_block}\n\n"
            f"Scene context: {ep_ctx}\n"
            f"Task: {goals[episode]}"
        )
        scene_intro = intros[episode]
    else:
        intros = {
            2: "*(Trzy dni później, zadymione zaplecze karczmy „Pod Złamanym Kołem”)*",
            3: "*(Tydzień później, świt nad Wąwozem Mgły. W dole widać blokadę wojsk Namiestnika)*"
        }
        goals = {
            2: "Rozpocznij spotkanie w karczmie dotyczące rozliczenia sprawy dziennika Torina i patroli Namiestnika, OD RAZU nawiązując konkretnie do tego, jak Kaelen zachował się wobec Ciebie pod Bramą Przełęczy Kruków. Napisz WYŁĄCZNIE bezpośrednią wypowiedź Varga (2-3 naturalne zdania po polsku, bez nawiasów).",
            3: "Kaelen przyszedł prosić o zbrojne wsparcie Czarnych Wilków do przełamania blokady w Wąwozie Mgły. Przywitaj go, wprost wypominając mu wcześniejsze zniewagi/oszustwa ALBO doceniając wcześniejszą pomoc i dotrzymane słowo. Napisz WYŁĄCZNIE bezpośrednią wypowiedź Varga (2-3 naturalne zdania po polsku, bez nawiasów)."
        }
        prompt = (
            f"{persona}\n\n"
            f"=== STAN EMOCJONALNY I RELACJA ===\n{relationship.to_Prompt_string('pl')}\n\n"
            f"=== WSPOMNIENIA Z POPRZEDNICH EPIZODÓW (PAMIĘĆ RAG) ===\n{memories_block}\n\n"
            f"Kontekst sceny: {ep_ctx}\n"
            f"Zadanie: {goals[episode]}"
        )
        scene_intro = intros[episode]

    try:
        reply = clean_stage_directions(llm.invoke([SystemMessage(content=prompt)]).content)
        return f"{scene_intro} {reply}"
    except Exception:
        return lang_openings[episode]


def generate_npc_response(
    mode: Literal["A", "B"],
    episode: int,
    chat_history: list,
    player_text: str,
    retrieved_memories: list[str] = None,
    relationship_state: str = "",
    lang: str = "pl",
    turn_number: int = 1,
    return_trace: bool = False
):
    """
    Generator odpowiedzi oparty na wzorcu ReAct (Reason + Act) + wyborze Best-of-5
    z mechanizmem naturalnego zamykania epizodu (end_episode) i twardymi widełkami tur (min 3, max 6).
    """
    default_end = _apply_end_guardrail(False, turn_number)
    empty_trace = {
        "method": "mock" if IS_MOCK_MODE else "fallback",
        "candidates": [],
        "reasoning": "",
        "rationale": "",
        "end_episode": default_end,
    }

    if IS_MOCK_MODE:
        reply = f"**[TRYB TESTOWY OFFLINE – Tryb {mode}, Epizod {episode}, Tura {turn_number}]** Varg patrzy na Ciebie uważnie."
        return (reply, empty_trace) if return_trace else reply

    persona = NPC_BASE_PERSONAS.get(lang, NPC_BASE_PERSONAS["pl"])
    ep_ctx = EPISODE_CONTEXTS.get(lang, EPISODE_CONTEXTS["pl"])[episode]

    history_lines = []
    for msg in chat_history:
        clean_msg = clean_stage_directions(msg["content"])
        speaker = "PLAYER (Kaelen)" if (lang == "en" and msg["role"] == "user") else (
            "GRACZ (Kaelen)" if msg["role"] == "user" else "VARG"
        )
        history_lines.append(f"{speaker}: {clean_msg}")
    history_text = "\n".join(history_lines) if history_lines else ("(Start of episode)" if lang == "en" else "(Początek epizodu)")

    # Reguła kończenia epizodu uzależniona od numeru tury
    if lang == "en":
        if turn_number < 3:
            pacing_rule = (
                f"PACING RULE (Turn {turn_number} of max 6): It is TOO EARLY to end this episode. "
                f"You MUST set `end_episode` to `false` and keep the conversation moving forward."
            )
        elif turn_number < 6:
            pacing_rule = (
                f"PACING RULE (Turn {turn_number} of max 6): You MAY set `end_episode` to `true` ONLY IF the scene's core matter is clearly resolved "
                f"(a concrete deal was struck, a final refusal/ultimatum was delivered, or Varg's patience ran out and he dismisses Kaelen). "
                f"If you set `end_episode` to `true`, `best_response` MUST be a decisive parting/closing line with NO open question at the end. "
                f"If negotiation is still active, keep `end_episode` set to `false`."
            )
        else:
            pacing_rule = (
                f"PACING RULE (FINAL TURN {turn_number}/6): This encounter MUST conclude right now. "
                f"Deliver a decisive, natural closing/parting statement from Varg summarizing the outcome (deal, warning, or dismissal — NO open questions) "
                f"and set `end_episode` to `true`."
            )

        nemesis_block = ""
        if mode == "B":
            memories_block = "\n".join([f"- {m}" for m in retrieved_memories]) if retrieved_memories else "No prior memories."
            nemesis_block = (
                f"\n\n=== CURRENT EMOTIONAL STATE & RELATIONSHIP (MODE B) ===\n{relationship_state}\n\n"
                f"=== RETRIEVED LONG-TERM MEMORIES FROM PAST EPISODES (RAG) ===\n{memories_block}\n"
                f"NEMESIS RULE: If past memories exist above (especially insults, threats, lies, promises, or help), weave them naturally into your response!"
            )

        react_instruction = (
            f"{persona}\n\n"
            f"=== CURRENT EPISODE CONTEXT ===\n{ep_ctx}"
            f"{nemesis_block}\n\n"
            f"=== CONVERSATION SO FAR IN THIS EPISODE ===\n{history_text}\n\n"
            f"=== {pacing_rule} ===\n\n"
            f"=== REACT + BEST-OF-5 INTERNAL PROCESS ===\n"
            f"Before answering Kaelen's latest message, execute a strict ReAct loop in JSON:\n"
            f"1. reasoning_analysis: Diagnose what the player actually proposed/did, identify what Varg ALREADY said above so you NEVER repeat the same phrases or ping-pong loop, and decide whether the scene should continue or conclude per the PACING RULE.\n"
            f"2. Generate 5 genuinely DIFFERENT spoken reply candidates in natural English (2-4 sentences each, NO brackets/asterisks): pragmatic, probing, pressure/ultimatum, narrative push, and personal/relational.\n"
            f"3. selection_rationale: Critique the 5 candidates and pick the single strongest, most natural, non-repetitive line.\n"
            f"4. best_response: Output the final winning spoken reply of Varg.\n"
            f"5. end_episode: Boolean flag strictly following the PACING RULE above."
        )
        user_turn_prompt = f"Player (Kaelen) just said (Turn {turn_number}): \"{player_text}\"\nGenerate the ReAct Best-of-5 structured response in English."
    else:
        if turn_number < 3:
            pacing_rule = (
                f"REGUŁA TEMPA SCENY (Tura {turn_number} z maks. 6): Jest ZBYT WCZEŚNIE, by kończyć ten epizod. "
                f"MUSISZ ustawić `end_episode` na `false` i prowadzić rozmowę dalej."
            )
        elif turn_number < 6:
            pacing_rule = (
                f"REGUŁA TEMPA SCENY (Tura {turn_number} z maks. 6): Możesz ustawić `end_episode` na `true` TYLKO WTEDY, gdy główny temat tego epizodu został wyraźnie rozstrzygnięty "
                f"(zawarto konkretną umowę, padła ostateczna odmowa/ultimatum albo Vargowi skończyła się cierpliwość i odprawia Kaelena). "
                f"Jeśli ustawiasz `end_episode` na `true`, `best_response` MUSI brzmieć jak stanowcze zakończenie spotkania (BEZ nowego pytania na końcu!). "
                f"Jeśli negocjacje nadal trwają, zostaw `end_episode` jako `false`."
            )
        else:
            pacing_rule = (
                f"REGUŁA TEMPA SCENY (OSTATNIA TURA {turn_number}/6): To spotkanie MUSI się teraz naturalnie zakończyć. "
                f"Napisz stanowczą kwestię zamykającą ze strony Varga podsumowującą wynik rozmowy (umowa, odprawa lub ostrzeżenie — BEZ otwartego pytania na końcu) "
                f"i ustaw `end_episode` na `true`."
            )

        nemesis_block = ""
        if mode == "B":
            memories_block = "\n".join([f"- {m}" for m in retrieved_memories]) if retrieved_memories else "Brak wcześniejszych wspomnień."
            nemesis_block = (
                f"\n\n=== AKTUALNY STAN EMOCJONALNY I RELACJA (TRYB B) ===\n{relationship_state}\n\n"
                f"=== ODZYSKANE WSPOMNIENIA Z POPRZEDNICH EPIZODÓW (PAMIĘĆ RAG) ===\n{memories_block}\n"
                f"ZASADA NEMESIS: Jeśli powyżej znajdują się wspomnienia z poprzednich epizodów (zwłaszcza zniewagi, groźby, kłamstwa, obietnice lub pomoc), bezpośrednio i naturalnie nawiąż do nich w swojej odpowiedzi!"
            )

        react_instruction = (
            f"{persona}\n\n"
            f"=== KONTEKST BIEŻĄCEGO EPIZODU ===\n{ep_ctx}"
            f"{nemesis_block}\n\n"
            f"=== DOTYCHCZASOWA ROZMOWA W TYM EPIZODZIE ===\n{history_text}\n\n"
            f"=== {pacing_rule} ===\n\n"
            f"=== PROCES WEWNĘTRZNY: PĘTLA ReAct + WYBÓR BEST-OF-5 ===\n"
            f"Zanim odpowiesz na najnowszą wypowiedź gracza (Kaelena), przeprowadź rozumowanie w JSON:\n"
            f"1. reasoning_analysis: Przeanalizuj dokładnie, co gracz zaproponował lub zrobił, wypisz czego Varg NIE MOŻE powtórzyć z powyższych tur (całkowity zakaz mantry i efektu ping-ponga) oraz oceń zgodnie z REGUŁĄ TEMPA SCENY, czy należy zamknąć ten epizod.\n"
            f"2. Wygeneruj 5 wyraźnie RÓŻNYCH wariantów odpowiedzi Varga w naturalnej, czystej polszczyźnie (po 2-4 zdania, ZERO nawiasów i gwiazdek): pragmatyczny, dociekliwy, stanowczy/ultimatum, rozwijający fabułę oraz osobisto-relacyjny.\n"
            f"3. selection_rationale: Porównaj 5 kandydatów i wskaż jednego, który brzmi najbardziej ludzko, nie powtarza poprzednich kwestii i najlepiej rozwija lub domyka rozmowę.\n"
            f"4. best_response: Wpisz finalną, zwycięską wypowiedź Varga po polsku.\n"
            f"5. end_episode: Wartość logiczna (true/false) zgodna z REGUŁĄ TEMPA SCENY."
        )
        user_turn_prompt = f"Gracz (Kaelen) właśnie powiedział (Tura {turn_number}): \"{player_text}\"\nWykonaj pętlę ReAct Best-of-5 i zwróć ustrukturyzowany wynik po polsku."

    # Próba 1: Natywny structured output (Pydantic ReActBestOf5Output)
    try:
        structured_react_llm = llm_react.with_structured_output(ReActBestOf5Output)
        result: ReActBestOf5Output = structured_react_llm.invoke([
            SystemMessage(content=react_instruction),
            HumanMessage(content=user_turn_prompt),
        ])
        final_reply = clean_stage_directions(result.best_response)
        if final_reply:
            should_end = _apply_end_guardrail(result.end_episode, turn_number)
            trace = {
                "method": "ReAct + Best-of-5 (Pydantic)",
                "reasoning": result.reasoning_analysis,
                "candidates": [
                    f"1 (Pragmatyczny): {clean_stage_directions(result.candidate_1_pragmatic)}",
                    f"2 (Dociekliwy): {clean_stage_directions(result.candidate_2_probing)}",
                    f"3 (Stanowczy): {clean_stage_directions(result.candidate_3_pressure)}",
                    f"4 (Fabularny): {clean_stage_directions(result.candidate_4_narrative_push)}",
                    f"5 (Relacyjny): {clean_stage_directions(result.candidate_5_personal)}",
                ],
                "rationale": result.selection_rationale,
                "end_episode": should_end,
            }
            return (final_reply, trace) if return_trace else final_reply
    except Exception:
        # Próba 2 (ADR-03): Zapasowy parser surowego JSON dla pętli ReAct Best-of-5
        try:
            json_schema_hint = (
                '\nOdpowiedz WYŁĄCZNIE poprawnym obiektem JSON z kluczami: '
                '"reasoning_analysis", "candidate_1_pragmatic", "candidate_2_probing", '
                '"candidate_3_pressure", "candidate_4_narrative_push", "candidate_5_personal", '
                '"selection_rationale", "best_response", "end_episode".'
            )
            raw = llm_react.invoke([
                SystemMessage(content=react_instruction + json_schema_hint),
                HumanMessage(content=user_turn_prompt),
            ]).content.strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            parsed = ReActBestOf5Output(**data)
            final_reply = clean_stage_directions(parsed.best_response)
            if final_reply:
                should_end = _apply_end_guardrail(parsed.end_episode, turn_number)
                trace = {
                    "method": "ReAct + Best-of-5 (JSON Fallback)",
                    "reasoning": parsed.reasoning_analysis,
                    "candidates": [
                        f"1 (Pragmatyczny): {clean_stage_directions(parsed.candidate_1_pragmatic)}",
                        f"2 (Dociekliwy): {clean_stage_directions(parsed.candidate_2_probing)}",
                        f"3 (Stanowczy): {clean_stage_directions(parsed.candidate_3_pressure)}",
                        f"4 (Fabularny): {clean_stage_directions(parsed.candidate_4_narrative_push)}",
                        f"5 (Relacyjny): {clean_stage_directions(parsed.candidate_5_personal)}",
                    ],
                    "rationale": parsed.selection_rationale,
                    "end_episode": should_end,
                }
                return (final_reply, trace) if return_trace else final_reply
        except Exception:
            pass

    # Próba 3: Bezpieczny fallback bezpośredni
    closing_hint = ""
    if turn_number >= 6:
        closing_hint = (
            "\nFINAL TURN: Wrap up the meeting decisively right now without asking new questions."
            if lang == "en"
            else "\nOSTATNIA TURA: Zakończ stanowczo to spotkanie bez zadawania nowych pytań."
        )
    fallback_system = f"{persona}\n\n{ep_ctx}{nemesis_block}{closing_hint}"
    messages = [SystemMessage(content=fallback_system)]
    for msg in chat_history:
        clean_msg_content = clean_stage_directions(msg["content"])
        if msg["role"] == "user":
            messages.append(HumanMessage(content=clean_msg_content))
        else:
            messages.append(AIMessage(content=clean_msg_content))
    messages.append(HumanMessage(content=player_text))

    response = llm.invoke(messages)
    fallback_reply = clean_stage_directions(response.content)
    return (fallback_reply, empty_trace) if return_trace else fallback_reply