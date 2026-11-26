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


class LLMGroq(LLM):
    """Groq : API compatible OpenAI (https://console.groq.com). Clé dans la variable GROQ_API_KEY."""

    def __init__(self, modele: str, max_tokens: int = 700, temperature: float = 0.2, delai: float = 60,
                 url: str = "https://api.groq.com/openai/v1"):
        cle = os.environ.get("GROQ_API_KEY")
        if not cle:
            raise ErreurLLM("Variable d'environnement GROQ_API_KEY absente.")
        self._cle, self.url = cle, url.rstrip("/")
        self.modele, self.max_tokens, self.temperature, self.delai = modele, max_tokens, temperature, delai

    def generer(self, systeme, utilisateur):
        import httpx
        corps = {
            "model": self.modele,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": systeme}, {"role": "user", "content": utilisateur}],
        }
        try:
            rep = httpx.post(f"{self.url}/chat/completions", json=corps, timeout=self.delai,
                             headers={"Authorization": f"Bearer {self._cle}"})
            rep.raise_for_status()
            texte = rep.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            raise ErreurLLM(f"Groq a refusé la requête (code {e.response.status_code}) : {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise ErreurLLM(f"Groq injoignable ou réponse invalide : {e}") from e
        if not texte:
            raise ErreurLLM("Réponse vide du modèle.")
        return texte


def creer_llm(config_llm: dict) -> LLM:
    f = config_llm["fournisseur"]
    t, d = config_llm.get("temperature", 0.2), config_llm.get("delai_secondes", 60)
    if f == "anthropic":
        c = config_llm["anthropic"]
        return LLMAnthropic(c["modele"], c.get("max_tokens", 700), t, d)
    if f == "groq":
        c = config_llm["groq"]
        return LLMGroq(c["modele"], c.get("max_tokens", 700), t, d)
    raise ValueError(f"Fournisseur LLM inconnu : {f!r} (groq | anthropic)")
