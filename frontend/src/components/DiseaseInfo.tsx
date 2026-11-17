/** Fiche maladie en onglets compacts (Description / Symptômes / Conseils), accessibles au clavier. */
import { useRef, useState, type KeyboardEvent } from 'react'
import { MALADIES } from '../data/maladies'
import { useI18n, type Cle } from '../i18n'
import type { ClasseMaladie } from '../types'
import css from './DiseaseInfo.module.css'

const ONGLETS: { id: 'desc' | 'symptomes' | 'conseils'; cle: Cle }[] = [
  { id: 'desc', cle: 'tabDesc' },
  { id: 'symptomes', cle: 'tabSymptoms' },
  { id: 'conseils', cle: 'tabAdvice' },
]

export default function DiseaseInfo({ classe }: { classe: ClasseMaladie }) {
  const { t, lang } = useI18n()
  const [actif, setActif] = useState(0)
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const fiche = MALADIES[classe][lang]

  // Navigation aux flèches (inversée en RTL), Début / Fin
  const clavier = (e: KeyboardEvent) => {
    const sens = lang === 'ar' ? -1 : 1
    let i = actif
    if (e.key === 'ArrowRight') i = actif + sens
    else if (e.key === 'ArrowLeft') i = actif - sens
    else if (e.key === 'Home') i = 0
    else if (e.key === 'End') i = ONGLETS.length - 1
    else return
    e.preventDefault()
    i = (i + ONGLETS.length) % ONGLETS.length
    setActif(i)
    refs.current[i]?.focus()
  }

  return (
    <section className={css.fiche} aria-label={t('infoTitle', { name: t(`class.${classe}`) })}>
      {/* Mention toujours visible */}
      <p className={css.avertissement}>{t('indicative')}</p>
      <div className={css.onglets} role="tablist" aria-label={t('tabsLabel')} onKeyDown={clavier}>
        {ONGLETS.map((o, i) => (
          <button
            key={o.id}
            ref={(el) => { refs.current[i] = el }}
            id={`onglet-${o.id}`}
            role="tab"
            type="button"
            aria-selected={i === actif}
            aria-controls="panneau-fiche"
            tabIndex={i === actif ? 0 : -1}
            className={`${css.onglet} ${i === actif ? css.actif : ''}`}
            onClick={() => setActif(i)}
          >
            {t(o.cle)}
          </button>
        ))}
      </div>
      <div id="panneau-fiche" role="tabpanel" aria-labelledby={`onglet-${ONGLETS[actif].id}`} tabIndex={0} className={css.panneau}>
        {actif === 0 && <p>{fiche.description}</p>}
        {actif === 1 && <ul>{fiche.symptomes.map((s) => <li key={s}>{s}</li>)}</ul>}
        {actif === 2 && <ul>{fiche.conseils.map((s) => <li key={s}>{s}</li>)}</ul>}
      </div>
    </section>
  )
}
