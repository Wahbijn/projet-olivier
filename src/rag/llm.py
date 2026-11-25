"""Clients LLM interchangeables. Un client = une méthode `generer(systeme, utilisateur) -> str`."""
import os


class ErreurLLM(RuntimeError):
    """Erreur côté LLM (clé absente, service injoignable, réponse vide...)."""


class LLM:
    def generer(self, systeme: str, utilisateur: str) -> str:
        raise NotImplementedError


class LLMAnthropic(LLM):
    def __init__(self, modele: str, max_tokens: int = 700, temperature: float = 0.2, delai: float = 60):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ErreurLLM("Variable d'environnement ANTHROPIC_API_KEY absente.")
        try:
            import anthropic
        except ImportError as e:
            raise ErreurLLM("Le paquet « anthropic » n'est pas installé : pip install anthropic") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=delai)
        self.modele, self.max_tokens, self.temperature = modele, max_tokens, temperature

    def generer(self, systeme, utilisateur):
        try:
            rep = self._client.messages.create(
                model=self.modele,
                max_tokens=self.max_tokens,
                system=systeme,
                messages=[{"role": "user", "content": utilisateur}],
            )
        except self._anthropic.APIError as e:
            raise ErreurLLM(f"Erreur de l'API Anthropic : {e}") from e
        texte = "".join(b.text for b in rep.content if getattr(b, "type", "") == "text").strip()
        if not texte:
            raise ErreurLLM("Réponse vide du modèle.")
        return texte


class LLMOllama(LLM):
    def __init__(self, url: str, modele: str, temperature: float = 0.2, delai: float = 60):
        self.url, self.modele, self.temperature, self.delai = url.rstrip("/"), modele, temperature, delai

    def generer(self, systeme, utilisateur):
        import httpx
        corps = {
            "model": self.modele,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": systeme}, {"role": "user", "content": utilisateur}],
        }
        try:
            rep = httpx.post(f"{self.url}/api/chat", json=corps, timeout=self.delai)
            rep.raise_for_status()
            texte = rep.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise ErreurLLM(f"Ollama injoignable ou réponse invalide ({self.url}) : {e}") from e
        if not texte:
            raise ErreurLLM("Réponse vide du modèle.")
        return texte


def creer_llm(config_llm: dict) -> LLM:
    f = config_llm["fournisseur"]
    t, d = config_llm.get("temperature", 0.2), config_llm.get("delai_secondes", 60)
    if f == "anthropic":
        c = config_llm["anthropic"]
        return LLMAnthropic(c["modele"], c.get("max_tokens", 700), t, d)
    if f == "ollama":
        c = config_llm["ollama"]
        return LLMOllama(c["url"], c["modele"], t, d)
    raise ValueError(f"Fournisseur LLM inconnu : {f!r} (anthropic | ollama)")
