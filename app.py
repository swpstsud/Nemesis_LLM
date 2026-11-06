import uuid
import streamlit as st
from engine import (
    analyze_player_input,
    generate_npc_response,
    NemesisVectorMemory,
    RelationshipEngine,
    EPISODE_CONTEXTS,
)

st.set_page_config(page_title="Nemesis-LLM Engine (Tydzień 2)", layout="centered")

# Inicjalizacja stanu sesji
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "episode" not in st.session_state:
    st.session_state.episode = 1
if "mode" not in st.session_state:
    st.session_state.mode = "B"  # Domyślnie Tryb B (Nemesis-LLM) do testów pamięci
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "perception_logs" not in st.session_state:
    st.session_state.perception_logs = []
if "last_rag_memories" not in st.session_state:
    st.session_state.last_rag_memories = []
if "vector_memory" not in st.session_state:
    st.session_state.vector_memory = NemesisVectorMemory(st.session_state.session_id)
if "relationship" not in st.session_state:
    st.session_state.relationship = RelationshipEngine()

# Pasek boczny dla badacza (Panel Debugowania)
with st.sidebar:
    st.header("Panel Badacza (Debug)")

    # Przełącznik Trybu A / B
    selected_mode = st.radio(
        "Tryb działania silnika (ZN):",
        options=["B", "A"],
        format_func=lambda x: "Tryb B: Nemesis-LLM (RAG + Afekt)" if x == "B" else "Tryb A: Vanilla LLM (Kontrolny)",
        index=0 if st.session_state.mode == "B" else 1
    )
    st.session_state.mode = selected_mode

    st.write(f"**Epizod:** {st.session_state.episode} / 3")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.episode < 3:
            if st.button("Następny epizod ⏭️"):
                st.session_state.episode += 1
                # Czyszczenie pamięci krótkotrwałej między epizodami!
                st.session_state.chat_history = []
                st.rerun()
    with col2:
        if st.button("Reset sesji 🔄"):
            st.session_state.episode = 1
            st.session_state.chat_history = []
            st.session_state.perception_logs = []
            st.session_state.last_rag_memories = []
            st.session_state.vector_memory.clear_all()
            st.session_state.relationship = RelationshipEngine()
            st.rerun()

    st.divider()
    st.subheader("Moduł Afektywny (Relationship Engine)")
    rel = st.session_state.relationship
    st.write(f"😡 **Wrogość:** {rel.wrogosc}/10 | 🤝 **Wdzięczność:** {rel.wdziecznosc}/10")
    st.write(f"🛡️ **Szacunek:** {rel.szacunek}/5 | ⚠️ **Lęk:** {rel.lek}/10")
    st.caption(rel.get_dominant_attitude())

    st.divider()
    st.subheader("Ostatnio pobrane z bazy RAG:")
    if st.session_state.mode == "A":
        st.info("Tryb A aktywny – baza wektorowa RAG jest wyłączona.")
    elif st.session_state.last_rag_memories:
        for mem in st.session_state.last_rag_memories:
            st.caption(f"🧠 {mem}")
    else:
        st.caption("Brak pobranych wspomnień.")

    st.divider()
    st.subheader("Logi Modułu Percepcji:")
    for log in st.session_state.perception_logs:
        st.caption(f"[Ep. {log['ep']}] **{log['cat']}** (Afekt: {log['score']}): {log['summary']}")

# Główny ekran gry
st.title("Nemesis-LLM Engine – Pamięć RAG i Afekt (Tydzień 2)")
st.subheader(f"Aktualny etap: {EPISODE_CONTEXTS[st.session_state.episode]}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if player_input := st.chat_input("Wpisz swoją wypowiedź do Varga..."):
    with st.chat_message("user"):
        st.markdown(player_input)

    with st.spinner("Varg analizuje Twoje słowa i przeszukuje pamięć..."):
        # 1. Moduł Percepcji analizuje wypowiedź
        perception = analyze_player_input(player_input)
        st.session_state.perception_logs.append({
            "ep": st.session_state.episode,
            "cat": perception.category,
            "score": perception.affect_score,
            "summary": perception.summary
        })

        # 2. Wyszukiwanie semantyczne RAG (przed dodaniem bieżącej wypowiedzi, aby pobrać historię)
        retrieved_memories = []
        if st.session_state.mode == "B":
            retrieved_memories = st.session_state.vector_memory.retrieve_relevant_memories(
                query_text=player_input,
                n_results=4
            )
            st.session_state.last_rag_memories = retrieved_memories

        # 3. Aktualizacja Modułu Afektywnego (Relationship Engine)
        st.session_state.relationship.update(perception)

        # 4. Generowanie odpowiedzi NPC (Tryb A lub Tryb B)
        npc_reply = generate_npc_response(
            mode=st.session_state.mode,
            episode=st.session_state.episode,
            chat_history=st.session_state.chat_history,
            player_text=player_input,
            retrieved_memories=retrieved_memories,
            relationship_state=st.session_state.relationship.to_Prompt_string()
        )

        # 5. Zapis nowej interakcji do bazy wektorowej ChromaDB (Pamięć Długotrwała)
        if st.session_state.mode == "B":
            st.session_state.vector_memory.add_memory(
                episode=st.session_state.episode,
                player_text=player_input,
                perception=perception
            )

    # Zapis do pamięci krótkotrwałej bieżącego epizodu
    st.session_state.chat_history.append({"role": "user", "content": player_input})
    st.session_state.chat_history.append({"role": "assistant", "content": npc_reply})
    st.rerun()