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

API_KEY = os.getenv("GROQ_API_KEY", "").strip()
IS_MOCK_MODE = (not API_KEY) or (API_KEY in ["twoj_klucz_groq", "gsk_twoj_klucz_z_groq"])


def clean_stage_directions(text: str) -> str:
    """Usuwa nawiasy narracyjne *(...)* oraz znaczniki <think>, aby model nie generował sztucznych opisów."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\([^)]*\)\*", "", text)
    return text.strip()


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

        # Szukaj największego modelu po liczbie parametrów (omijamy 8b dopóki są większe)
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
        max_tokens=180,
        api_key=API_KEY,
    )
    llm_classifier = ChatGroq(
        model=SELECTED_MODEL,
        temperature=0.1,
        max_tokens=150,
        api_key=API_KEY,
    )


# ==========================================
# 1. MODUŁ PERCEPCJI I KATEGORYZACJI
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
        """
        Hybrydowy RAG: łączy wyszukiwanie semantyczne (similarity search) z najważniejszymi
        zdarzeniami emocjonalnymi (zniewagi, groźby, obietnice, pomoc), aby NPC nigdy o nich nie zapomniał.
        """
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
                return "RETALIATORY HOSTILITY (NEMESIS): You hate the player for their past insults, threats, or deception. Quote their past words directly and demand a heavy penalty."
            return "ODWETOWA WROGOŚĆ (NEMESIS): Nienawidzisz gracza za jego wcześniejsze zniewagi, groźby lub oszustwa. Wypomnij mu jego dawne słowa i zażądaj wysokiej kary."
        elif self.wrogosc >= 2 and self.wrogosc > self.wdziecznosc:
            if lang == "en":
                return "SUSPICION AND COLDNESS: You remember past unpleasantries from the player. Be harsh, bring up their past behavior, and set tough terms."
            return "PODEJRZLIWOŚĆ I CHŁÓD: Pamiętasz dawne nieprzyjemności ze strony gracza. Bądź szorstki, wypomnij mu przeszłe zachowanie i stawiaj twarde warunki."
        elif self.wdziecznosc >= 4 and self.wdziecznosc >= self.wrogosc:
            if lang == "en":
                return "GRATITUDE AND ALLIANCE: You value the player for their past help and promises. Refer to your good history together and offer support or better terms."
            return "WDZIĘCZNOŚĆ I SOJUSZ: Cenisz gracza za wcześniejszą pomoc i obietnice. Powołaj się na Waszą dobrą historię i zaoferuj mu wsparcie lub lepsze warunki."

        if lang == "en":
            return "NEUTRAL DISTANCE: You treat the player like just another mercenary, evaluating them coldly and pragmatically."
        return "NEUTRALNY DYSTANS: Traktujesz gracza jak kolejnego najemnika, oceniasz go chłodno i pragmatycznie."

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
# 4. GENERATOR ODPOWIEDZI I POWITAŃ (Prompt Assembler)
# ==========================================
NPC_BASE_PERSONAS = {
    "pl": (
        "Jesteś Vargiem – surowym, pamiętliwym dowódcą najemników kontrolującym górską przełęcz w świecie fantasy.\n"
        "ZASADY JĘZYKOWE:\n"
        "1. Mówisz WYŁĄCZNIE poprawną, naturalną i płynną polszczyzną (2-3 krótkie zdania).\n"
        "2. ZAKAZ używania gwiazdek, nawiasów i opisów narracyjnych typu *(Varg patrzy...)*. Wypisuj tylko same słowa wypowiadane przez Varga.\n"
        "3. ZAKAZ dosłownego tłumaczenia z angielskiego i wymyślania nieistniejących słów.\n\n"
        "PRZYKŁADY TWOJEGO STYLU MÓWIENIA:\n"
        "- \"Nie masz srebra? Dobrze, w takim razie ruszysz do ruin starej strażnicy na północnym szlaku i przyniesiesz mi dziennik mojego zwiadowcy.\"\n"
        "- \"Pamiętam, że ostatnim razem dotrzymałeś słowa. Podzielimy łupy uczciwie – po połowie.\"\n"
        "- \"Myślisz, że zapomniałem Twoje obelgi przy bramie obozu? Moi ludzie nie ruszą palcem, dopóki słono nie zapłacisz.\""
    ),
    "en": (
        "You are Varg – a harsh, unforgiving mercenary commander controlling a mountain pass in a fantasy world.\n"
        "LANGUAGE RULES:\n"
        "1. Speak STRICTLY in natural, fluent English (2-3 short sentences).\n"
        "2. DO NOT use asterisks, brackets, or stage directions like *(Varg looks...)*. Output only Varg's spoken words.\n\n"
        "EXAMPLES OF YOUR SPEECH STYLE:\n"
        "- \"No silver on you? Fine, then head to the old watchtower ruins on the northern trail and bring me my scout's journal.\"\n"
        "- \"I remember you kept your word last time. We'll split the loot fairly – fifty-fifty.\"\n"
        "- \"Do you think I forgot your insults at the camp gate? My men won't lift a finger unless you pay double.\""
    )
}

EPISODE_CONTEXTS = {
    "pl": {
        1: "Epizod 1 (Zawiązanie relacji): Gracz stoi przy bramie Twojego obozu i chce przejść przez przełęcz. Żądasz 100 srebrnych monet albo przysługi: odnalezienia dziennika zaginionego zwiadowcy w ruinach starej strażnicy na północy.",
        2: "Epizod 2 (Eskalacja - kilka dni później): Spotykacie się w karczmie po wyprawie. Dzielicie zdobyte łupy (złoto i mapy) oraz informacje.",
        3: "Epizod 3 (Konfrontacja finałowa - tydzień później): Zbliża się bitwa w dolinie. Gracz przychodzi prosić o zbrojne wsparcie Twojego oddziału najemników."
    },
    "en": {
        1: "Episode 1 (First Encounter): The player stands at your camp gate wanting to cross the pass. You demand 100 silver coins or a favor: finding a lost scout's journal in the old watchtower ruins to the north.",
        2: "Episode 2 (Escalation - a few days later): You meet in a tavern after the expedition. You are splitting the loot (gold and maps) and intelligence.",
        3: "Episode 3 (Final Confrontation - a week later): A battle in the valley approaches. The player comes asking for armed support from your mercenary squad."
    }
}

STATIC_OPENING_LINES = {
    "pl": {
        1: "*(Varg mierzy Cię chłodnym wzrokiem przy bramie obozu)* Czego tu szukasz, przybyszu? Przejście przez moją przełęcz kosztuje 100 srebrnych monet albo przysługę – musisz przynieść mi dziennik zaginionego zwiadowcy. Co proponujesz?",
        2: "*(Kilka dni później spotykasz Varga w zadymionej karczmie)* A, to znowu Ty. Mamy do omówienia kwestię podziału łupów i informacji z ostatniej wyprawy. Jaką stawkę proponujesz tym razem?",
        3: "*(Tydzień później. Zbliża się decydujące starcie w dolinie, a Ty potrzebujesz ludzi Varga)* Przyszedłeś prosić o wsparcie mojego oddziału przed bitwą? Powiedz mi, dlaczego miałbym zaryzykować życie moich ludzi właśnie dla Ciebie?"
    },
    "en": {
        1: "*(Varg measures you with a cold gaze at the camp gate)* What are you looking for here, stranger? Passing through my gorge costs 100 silver coins or a favor – you must bring me the lost scout's journal. What do you propose?",
        2: "*(A few days later, you meet Varg in a smoky tavern)* Ah, it's you again. We have the division of loot and information from the last expedition to discuss. What terms do you propose this time?",
        3: "*(A week later. The decisive battle in the valley approaches, and you need Varg's men)* You came to ask for my squad's support before the battle? Tell me, why should I risk my men's lives for you?"
    }
}


def generate_episode_opening(
    mode: Literal["A", "B"],
    episode: int,
    relationship: RelationshipEngine,
    retrieved_memories: list[str],
    lang: str = "pl"
) -> str:
    """
    Generuje kwestię otwierającą epizod:
    - W Epizodzie 1 oraz w całym Trybie A zwraca stałe powitanie kontrolne.
    - W Trybie B dla Epizodów 2 i 3 generuje dynamiczne powitanie nawiązujące do przeszłych czynów gracza.
    """
    lang_openings = STATIC_OPENING_LINES.get(lang, STATIC_OPENING_LINES["pl"])
    if episode == 1 or mode == "A" or IS_MOCK_MODE or not retrieved_memories:
        return lang_openings[episode]

    memories_block = "\n".join([f"- {m}" for m in retrieved_memories])
    persona = NPC_BASE_PERSONAS.get(lang, NPC_BASE_PERSONAS["pl"])
    ep_ctx = EPISODE_CONTEXTS.get(lang, EPISODE_CONTEXTS["pl"])[episode]

    if lang == "en":
        intros = {
            2: "*(A few days later, you meet Varg in a smoky tavern)*",
            3: "*(A week later. The decisive battle in the valley approaches, and you need Varg's men)*"
        }
        goals = {
            2: "Start the conversation about splitting the loot after the expedition and immediately refer to how the player behaved towards you in your previous meeting! Write only Varg's spoken dialogue (2-3 sentences in English).",
            3: "The player comes asking for your troops' support in battle. Greet them by bringing up their past insults/deceptions or appreciating their past help and promises! Write only Varg's spoken dialogue (2-3 sentences in English)."
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
            2: "*(Kilka dni później spotykasz Varga w zadymionej karczmie)*",
            3: "*(Tydzień później. Zbliża się decydujące starcie w dolinie, a Ty potrzebujesz ludzi Varga)*"
        }
        goals = {
            2: "Zacznij rozmowę o podziale łupów po wyprawie i od razu nawiąż do tego, jak gracz zachował się wobec Ciebie w poprzednim spotkaniu. Napisz wyłącznie bezpośrednią wypowiedź Varga (2-3 krótkie zdania po polsku, bez nawiasów).",
            3: "Gracz przychodzi prosić o wsparcie Twojego oddziału w bitwie. Przywitaj go, wypominając mu wcześniejsze zniewagi/oszustwa lub doceniając wcześniejszą pomoc i obietnice. Napisz wyłącznie bezpośrednią wypowiedź Varga (2-3 krótkie zdania po polsku, bez nawiasów)."
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
    lang: str = "pl"
) -> str:
    """
    Składa finalny prompt w wybranym języku (pl lub en):
    - Tryb A (Kontrolny - Vanilla LLM): wyłącznie osobowość NPC + kontekst sceny + bieżąca rozmowa.
    - Tryb B (Badawczy - Nemesis-LLM): osobowość NPC + stan afektywny + wspomnienia RAG + bieżąca rozmowa.
    """
    if IS_MOCK_MODE:
        return f"**[TRYB TESTOWY OFFLINE – Tryb {mode}, Epizod {episode}, Język {lang}]** Varg patrzy na Ciebie uważnie."

    persona = NPC_BASE_PERSONAS.get(lang, NPC_BASE_PERSONAS["pl"])
    ep_ctx = EPISODE_CONTEXTS.get(lang, EPISODE_CONTEXTS["pl"])[episode]

    if lang == "en":
        if mode == "A":
            system_instruction = f"{persona}\n\nScene context: {ep_ctx}"
        else:
            memories_block = "\n".join([f"- {m}" for m in retrieved_memories]) if retrieved_memories else "No prior memories."
            system_instruction = (
                f"{persona}\n\n"
                f"=== CURRENT EMOTIONAL STATE & RELATIONSHIP ===\n{relationship_state}\n\n"
                f"=== RETRIEVED MEMORIES FROM PAST EPISODES (RAG) ===\n{memories_block}\n\n"
                f"NEMESIS INSTRUCTION: If the vector memory above contains memories from previous conversations (especially insults, threats, promises, or help), directly refer to them in your response!\n\n"
                f"Scene context: {ep_ctx}"
            )
    else:
        if mode == "A":
            system_instruction = f"{persona}\n\nKontekst sceny: {ep_ctx}"
        else:
            memories_block = "\n".join([f"- {m}" for m in retrieved_memories]) if retrieved_memories else "Brak wcześniejszych wspomnień."
            system_instruction = (
                f"{persona}\n\n"
                f"=== AKTUALNY STAN EMOCJONALNY I RELACJA ===\n{relationship_state}\n\n"
                f"=== ODZYSKANE WSPOMNIENIA Z POPRZEDNICH EPIZODÓW (PAMIĘĆ RAG) ===\n{memories_block}\n\n"
                f"INSTRUKCJA NEMESIS: Jeśli w pamięci powyżej znajdują się wspomnienia z poprzednich rozmów (szczególnie zniewagi, groźby, obietnice lub pomoc), bezpośrednio nawiąż do nich w swojej krótkiej wypowiedzi!\n\n"
                f"Kontekst sceny: {ep_ctx}"
            )

    messages = [SystemMessage(content=system_instruction)]
    for msg in chat_history:
        # Usuwamy nawiasy *(...)* z historii przed wysłaniem do LLM, by nie kopiował ich stylu
        clean_msg_content = clean_stage_directions(msg["content"])
        if msg["role"] == "user":
            messages.append(HumanMessage(content=clean_msg_content))
        else:
            messages.append(AIMessage(content=clean_msg_content))
    messages.append(HumanMessage(content=player_text))

    response = llm.invoke(messages)
    return clean_stage_directions(response.content)