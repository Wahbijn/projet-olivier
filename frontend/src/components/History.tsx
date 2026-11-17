/** Historique de la session : vignettes cliquables des dernières analyses (mémoire uniquement). */
import { useI18n } from '../i18n'
import type { Analyse } from '../types'
import css from './History.module.css'

interface Props {
  analyses: Analyse[]
  activeId: number | null
  onSelect: (a: Analyse) => void
}

export default function History({ analyses, activeId, onSelect }: Props) {
  const { t } = useI18n()

  return (
    <section className={css.historique} aria-labelledby="titre-historique">
      <h2 id="titre-historique" className={css.titre}>{t('historyTitle')}</h2>
      {analyses.length === 0 ? (
        <p className={css.vide}>{t('historyEmpty')}</p>
      ) : (
        <ul className={css.liste}>
          {analyses.map((a) => {
            const nom = t(`class.${a.resultat.classe}`)
            return (
              <li key={a.id}>
                <button
                  type="button"
                  className={`${css.vignette} ${a.id === activeId ? css.active : ''}`}
                  style={{ ['--couleur' as string]: `var(--c-${a.resultat.classe})` }}
                  aria-label={t('historyItem', { n: a.id, name: nom })}
                  title={nom}
                  aria-current={a.id === activeId ? 'true' : undefined}
                  onClick={() => onSelect(a)}
                >
                  <img src={a.photoPreparee} alt="" />
                  <span className={css.etiquette}>{t(`short.${a.resultat.classe_predite}`)}{a.resultat.incertain ? ' ?' : ''}</span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
      <p className={css.note}>{t('historyNote')}</p>
    </section>
  )
}
