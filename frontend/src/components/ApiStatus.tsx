/** Pastille d'état de l'API : interroge /health au chargement, puis toutes les 30 s. */
import { useEffect, useState } from 'react'
import { lireSante } from '../api'
import { useI18n } from '../i18n'
import css from './ApiStatus.module.css'

type Etat = 'verification' | 'ok' | 'degrade' | 'injoignable'

export default function ApiStatus() {
  const { t } = useI18n()
  const [etat, setEtat] = useState<Etat>('verification')

  useEffect(() => {
    let actif = true
    const verifier = async () => {
      try {
        const s = await lireSante()
        if (actif) setEtat(s.statut === 'ok' && s.modele_charge ? 'ok' : 'degrade')
      } catch {
        if (actif) setEtat('injoignable')
      }
    }
    verifier()
    const id = setInterval(verifier, 30_000)
    return () => {
      actif = false
      clearInterval(id)
    }
  }, [])

  const libelle = {
    verification: t('apiChecking'),
    ok: t('apiOk'),
    degrade: t('apiDegraded'),
    injoignable: t('apiDown'),
  }[etat]

  return (
    <div className={css.statut} role="status">
      {/* La couleur n'est jamais le seul indicateur : le texte l'accompagne */}
      <span className={`${css.pastille} ${css[etat]}`} aria-hidden="true" />
      <span>{libelle}</span>
    </div>
  )
}
