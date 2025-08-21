import uuid
from datetime import datetime

class MemoryManager:
    def __init__(self):
        self.sessions = {}

    def _generate_session_id(self):
        return str(uuid.uuid4())

    def add_interaction(self, session_id, query, response):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        interaction = {
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow()
        }
        self.sessions[session_id].append(interaction)

    def get_recent_context(self, session_id, n=5, relevance_keywords=None):
        if session_id not in self.sessions:
            return []

        interactions = self.sessions[session_id]

        if relevance_keywords:
            relevant_interactions = [
                inter for inter in interactions
                if any(keyword in inter["query"] or keyword in inter["response"] for keyword in relevance_keywords)
            ]
        else:
            relevant_interactions = interactions

        return relevant_interactions[-n:]

    def clear_memory(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def start_session(self):
        session_id = self._generate_session_id()
        self.sessions[session_id] = []
        return session_id
