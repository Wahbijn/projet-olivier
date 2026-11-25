/** Assistant agronomique (RAG) : questions libres ou suggérées, réponses avec sources citées. */
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError, poserQuestion, ragPret } from '../api'
import { QUESTIONS } from '../data/questions'
import { useI18n, type Cle } from '../i18n'
import type { Analyse, RagSource } from '../types'
import css from './Assistant.module.css'

interface Message {
  id: number
  role: 'user' | 'bot'
  texte: string
  sources?: RagSource[]
  refuse?: boolean
}

const LONGUEUR_MAX = 500 // identique à la validation de l'API

/** Les liens ne sont ouverts que s'ils sont en http(s) (jamais javascript:, data:...). */
const lienSur = (url: string | null) => (url && /^https?:\/\//i.test(url) ? url : null)

export default function Assistant({ analyse }: { analyse: Analyse }) {
  const { t, lang } = useI18n()
  const [pret, setPret] = useState<boolean | null>(null) // null = vérification en cours
  const [messages, setMessages] = useState<Message[]>([])
  const [saisie, setSaisie] = useState('')
  const [enCours, setEnCours] = useState(false)
  const [erreurCle, setErreurCle] = useState<Cle | null>(null)
  const compteur = useRef(0)
  const liste = useRef<HTMLDivElement>(null)

  const { classe, confiance } = analyse.resultat
  const suggestions = QUESTIONS[classe][lang]

  useEffect(() => {
    let actif = true
    ragPret().then((ok) => actif && setPret(ok))
    return () => { actif = false }
  }, [])

  // Fait défiler la conversation vers le dernier message
  useEffect(() => {
    liste.current?.scrollTo({ top: liste.current.scrollHeight, behavior: 'smooth' })
  }, [messages, enCours])

  const envoyer = async (texte: string) => {
    const question = texte.trim()
    if (!question || enCours) return
    setSaisie('')
    setErreurCle(null)
    setMessages((m) => [...m, { id: ++compteur.current, role: 'user', texte: question }])
    setEnCours(true)
    try {
      const rep = await poserQuestion(question, classe, confiance, lang)
      setMessages((m) => [
        ...m,
        { id: ++compteur.current, role: 'bot', texte: rep.reponse, sources: rep.sources, refuse: rep.refuse },
      ])
    } catch (e) {
      setErreurCle(e instanceof ApiError && e.type === 'injoignable' ? 'ragErrDown' : 'ragErrGeneric')
    } finally {
      setEnCours(false)
    }
  }

  const soumettre = (e: FormEvent) => {
    e.preventDefault()
    void envoyer(saisie)
  }

  if (pret === false) {
    return (
      <section className={css.assistant} aria-labelledby="titre-rag">
        <h2 id="titre-rag" className={css.titre}>{t('ragTitle')}</h2>
        <p className={css.indispo}>{t('ragUnavailable')}</p>
      </section>
    )
  }

  return (
    <section className={css.assistant} aria-labelledby="titre-rag">
      <h2 id="titre-rag" className={css.titre}>{t('ragTitle')}</h2>
      <p className={css.intro}>{t('ragIntro')}</p>

      {messages.length === 0 && (
        <div className={css.suggestions} role="group" aria-label={t('ragSuggestions')}>
          {suggestions.map((q) => (
            <button key={q} type="button" className={css.puce} onClick={() => void envoyer(q)} disabled={enCours || pret === null}>
              {q}
            </button>
          ))}
        </div>
      )}

      {(messages.length > 0 || enCours) && (
        <div ref={liste} className={css.conversation} role="log" aria-live="polite">
          {messages.map((m) => (
            <div key={m.id} className={`${css.message} ${m.role === 'user' ? css.utilisateur : css.bot}`}>
              <span className={css.auteur}>{m.role === 'user' ? t('ragYou') : t('ragBot')}</span>
              <p dir="auto" className={css.texte}>{m.texte}</p>
              {m.sources && m.sources.length > 0 && !m.refuse && (
                <details className={css.sources}>
                  <summary>{t('ragSources', { n: m.sources.length })}</summary>
                  <ol>
                    {m.sources.map((s) => {
                      const href = lienSur(s.url)
                      return (
                        <li key={s.numero} value={s.numero}>
                          <strong dir="auto">{s.titre}</strong>
                          <q dir="auto" className={css.extrait}>{s.extrait}</q>
                          {href && (
                            <a href={href} target="_blank" rel="noopener noreferrer">{t('ragOpenSource')}</a>
                          )}
                        </li>
                      )
                    })}
                  </ol>
                </details>
              )}
            </div>
          ))}
          {enCours && <p className={css.attente}>{t('ragThinking')}</p>}
        </div>
      )}

      {erreurCle && <p className={css.erreur} role="alert">{t(erreurCle)}</p>}

      <form className={css.formulaire} onSubmit={soumettre}>
        <label htmlFor="question-rag" className={css.etiquette}>{t('ragLabel')}</label>
        <input
          id="question-rag"
          type="text"
          value={saisie}
          maxLength={LONGUEUR_MAX}
          onChange={(e) => setSaisie(e.target.value)}
          placeholder={t('ragPlaceholder')}
          disabled={enCours || pret === null}
          autoComplete="off"
        />
        <button type="submit" className="btn" disabled={enCours || pret === null || saisie.trim().length < 3}>
          {t('ragSend')}
        </button>
      </form>

      <p className={css.mention}>{t('ragDisclaimer')}</p>
    </section>
  )
}
