import os
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


def get_active_groq_model(api_key: str) -> str:
    """Pobiera z API Groq listę aktywnych modeli i wybiera najlepszy model konwersacyjny Llama."""
    try:
        client = Groq(api_key=api_key)
        available_models = [m.id for m in client.models.list().data]
        preferred = [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ]
        for model_name in preferred:
            if model_name in available_models:
                return model_name

        excluded = ["whisper", "guard", "tts", "playai", "orpheus", "embed", "safeguard"]
        chat_models = [m for m in available_models if not any(ex in m.lower() for ex in excluded)]
        llama_models = [m for m in chat_models if "llama" in m.lower()]
        return llama_models[0] if llama_models else chat_models[0]
    except Exception:
        return "llama-3.1-8b-instant"


if not IS_MOCK_MODE:
    SELECTED_MODEL = get_active_groq_model(API_KEY)
    llm = ChatGroq(model=SELECTED_MODEL, temperature=0.7, api_key=API_KEY)


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
        description="Krótkie, jednozdaniowe podsumowanie po polsku tego, co gracz zrobił lub powiedział wobec NPC."
    )


def analyze_player_input(player_text: str) -> PerceptionOutput:
    """Klasyfikuje wypowiedź gracza pod kątem intencji i ładunku afektywnego."""
    if IS_MOCK_MODE:
        return PerceptionOutput(category="neutralne", affect_score=0, summary=f"Gracz powiedział: '{player_text}'")

    system_prompt = (
        "Jesteś modułem analitycznym w grze RPG. Przeanalizuj wypowiedź gracza skierowaną do NPC "
        "i zwróć wyłącznie obiekt JSON z trzema polami:\n"
        '- "category": jedna z wartości ["pomoc", "zniewaga", "groźba", "obietnica", "kłamstwo", "neutralne"]\n'
        '- "affect_score": liczba całkowita od -2 (wroga/groźba/zniewaga) do 2 (życzliwa/pomocna)\n'
        '- "summary": krótkie, jednozdaniowe podsumowanie po polsku co gracz zrobił wobec NPC.'
    )

    try:
        structured_llm = llm.with_structured_output(PerceptionOutput)
        return structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=player_text)
        ])
    except Exception:
        try:
            raw = llm.invoke([
                SystemMessage(content=system_prompt + "\nOdpowiedz TYLKO czystym kodem JSON."),
                HumanMessage(content=player_text)
            ]).content.strip()
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            return PerceptionOutput(**data)
        except Exception:
            return PerceptionOutput(category="neutralne", affect_score=0, summary=f"Gracz powiedział: '{player_text}'")


# ==========================================
# 2. MODUŁ PAMIĘCI WEKTOROWEJ (ChromaDB + RAG)
# ==========================================
class NemesisVectorMemory:
    """Zarządza bazą wektorową ChromaDB oraz wyszukiwaniem semantycznym RAG."""
    def __init__(self, session_id: str):
        self.client = chromadb.Client()
        self.collection_name = f"nemesis_mem_{session_id}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_memory(self, episode: int, player_text: str, perception: PerceptionOutput):
        """Zapisuje zdarzenie w bazie wektorowej z metadanymi (timestamp, epizod, kategoria, afekt)."""
        # Pomijamy zupełnie neutralne przywitania, by nie zaśmiecać pamięci długotrwałej
        doc_text = (
            f"Epizod {episode}: {perception.summary} "
            f"(Dokładne słowa gracza: \"{player_text}\", Kategoria: {perception.category})"
        )
        metadata = {
            "episode": episode,
            "category": perception.category,
            "affect_score": perception.affect_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_text": player_text
        }
        self.collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def retrieve_relevant_memories(self, query_text: str, n_results: int = 4) -> list[str]:
        """Mechanizm RAG: wyszukuje w bazie wektorowej najbardziej powiązane wspomnienia z przeszłości."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count)
        )
        return results["documents"][0] if results["documents"] else []

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

    def get_dominant_attitude(self) -> str:
        if self.wrogosc >= 5 and self.wrogosc > self.wdziecznosc:
            return (
                "ODWETOWA WROGOŚĆ (NEMESIS): Nienawidzisz gracza za jego wcześniejsze zniewagi, groźby lub oszustwa. "
                "Wypomnij mu dokładnie jego dawne słowa i szukaj okazji do zemsty lub zażądaj wysokiej kary."
            )
        elif self.wrogosc >= 2 and self.wrogosc > self.wdziecznosc:
            return (
                "PODEJRZLIWOŚĆ I CHŁÓD: Pamiętasz dawne nieprzyjemności ze strony gracza. "
                "Bądź szorstki, wypomnij mu przeszłe zachowanie i stawiaj twarde warunki."
            )
        elif self.wdziecznosc >= 4 and self.wdziecznosc >= self.wrogosc:
            return (
                "WDZIĘCZNOŚĆ I SOJUSZ: Cenisz gracza za wcześniejszą pomoc i obietnice. "
                "Powołaj się na Waszą dobrą historię i zaoferuj mu wsparcie lub lepsze warunki."
            )
        return "NEUTRALNY DYSTANS: Traktujesz gracza jak kolejnego najemnika, oceniasz go chłodno i pragmatycznie."

    def to_Prompt_string(self) -> str:
        return (
            f"Wrogość: {self.wrogosc}/10 | Wdzięczność: {self.wdziecznosc}/10 | "
            f"Szacunek: {self.szacunek}/5 | Lęk: {self.lek}/10\n"
            f"Dominująca postawa: {self.get_dominant_attitude()}"
        )


# ==========================================
# 4. GENERATOR ODPOWIEDZI (Prompt Assembler)
# ==========================================
NPC_BASE_PERSONA = """Jesteś Vargiem – wpływowym, pamiętliwym dowódcą najemników i handlarzem informacjami w mrocznym świecie fantasy.
Odpowiadasz wyłącznie po polsku, zwięźle (2-4 zdania), w naturalnym stylu dialogu z gry RPG.
Nie wychodź z roli i nie wspominaj, że jesteś sztuczną inteligencją."""

EPISODE_CONTEXTS = {
    1: "Epizod 1 (Zawiązanie relacji): Gracz przychodzi do Twojego obozu, prosząc o przepustkę przez przełęcz. Żądasz przysługi (dziennika zwiadowcy) lub zapłaty (100 srebrnych monet).",
    2: "Epizod 2 (Eskalacja - kilka dni później): Spotykacie się ponownie w karczmie. Negocjujecie podział łupów i informacji po wyprawie.",
    3: "Epizod 3 (Konfrontacja finałowa - tydzień później): Spotykacie się przed decydującym starciem. Gracz pilnie potrzebuje wsparcia Twoich ludzi."
}


def generate_npc_response(
    mode: Literal["A", "B"],
    episode: int,
    chat_history: list,
    player_text: str,
    retrieved_memories: list[str] = None,
    relationship_state: str = ""
) -> str:
    """
    Składa finalny prompt:
    - Tryb A (Kontrolny - Vanilla LLM): wyłącznie osobowość NPC + kontekst sceny + bieżąca rozmowa.
    - Tryb B (Badawczy - Nemesis-LLM): osobowość NPC + stan afektywny + wspomnienia RAG + bieżąca rozmowa.
    """
    if IS_MOCK_MODE:
        return f"**[TRYB TESTOWY OFFLINE – Tryb {mode}, Epizod {episode}]** Varg patrzy na Ciebie uważnie."

    if mode == "A":
        # TRYB A: Brak pamięci długotrwałej RAG i brak Relationship Engine
        system_instruction = (
            f"{NPC_BASE_PERSONA}\n"
            f"Kontekst bieżącej sceny: {EPISODE_CONTEXTS[episode]}"
        )
    else:
        # TRYB B: Pełna architektura Nemesis-LLM (RAG + Relationship Engine)
        memories_block = (
            "\n".join([f"- {m}" for m in retrieved_memories])
            if retrieved_memories else "Brak wcześniejszych wspomnień w bazie."
        )
        system_instruction = (
            f"{NPC_BASE_PERSONA}\n\n"
            f"=== AKTUALNY STAN EMOCJONALNY I RELACJA Z GRACZEM (Moduł Afektywny) ===\n"
            f"{relationship_state}\n\n"
            f"=== ODZYSKANE WSPOMNIENIA Z PRZESZŁOŚCI (Pamięć Wektorowa RAG) ===\n"
            f"{memories_block}\n\n"
            f"INSTRUKCJA NEMESIS: Jeśli w pamięci wektorowej powyżej znajdują się wspomnienia z poprzednich rozmów "
            f"(szczególnie zniewagi, groźby, obietnice lub pomoc), bezpośrednio nawiąż do nich w swojej wypowiedzi!\n\n"
            f"Kontekst bieżącej sceny: {EPISODE_CONTEXTS[episode]}"
        )

    messages = [SystemMessage(content=system_instruction)]
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=player_text))

    response = llm.invoke(messages)
    return response.content