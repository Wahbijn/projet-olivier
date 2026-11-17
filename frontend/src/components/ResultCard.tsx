/** Carte de résultat : diagnostic, jauge de confiance et probabilités. */
import { useI18n } from '../i18n'
import { CLASSES, type Analyse } from '../types'
import ConfidenceGauge from './ConfidenceGauge'
import { ClassIcon } from './Ornaments'
import css from './ResultCard.module.css'

export default function ResultCard({ analyse }: { analyse: Analyse }) {
  const { t } = useI18n()
  const r = analyse.resultat
  const incertain = r.classe === 'incertain'
  const couleur = `var(--c-${r.classe})`
  const seuilPct = Math.round(r.seuil_confiance * 100)

  return (
    // aria-live : le lecteur d'écran annonce l'arrivée du résultat
    <article className={css.carte} aria-live="polite" aria-labelledby="titre-resultat" key={analyse.id}>
      <p className={css.surtitre}>{t('resultTitle')}</p>

      <header className={css.entete} style={{ ['--couleur' as string]: couleur }}>
        <span className={css.icone}>
          <ClassIcon classe={r.classe} size={34} />
        </span>
        <h2 id="titre-resultat" className={css.diagnostic}>
          {incertain ? t('uncertainTitle') : t(`class.${r.classe}`)}
        </h2>
      </header>

      {incertain && (
        <div className={css.incertain}>
          <p>{t('uncertainText')}</p>
          <p className={css.indicatif}>
            <ClassIcon classe={r.classe_predite} size={22} /> {t(`class.${r.classe_predite}`)}
          </p>
          <h3>{t('uncertainAdviceTitle')}</h3>
          <ul>
            <li>{t('uncertainAdvice1')}</li>
            <li>{t('uncertainAdvice2')}</li>
            <li>{t('uncertainAdvice3')}</li>
          </ul>
        </div>
      )}

      <ConfidenceGauge confiance={r.confiance} seuil={r.seuil_confiance} couleur={couleur} />

      <section aria-labelledby="titre-probas" className={css.probas}>
        <h3 id="titre-probas" className={css.sousTitre}>{t('probabilities')}</h3>
        <ul className={css.barres}>
          {CLASSES.map((c, i) => {
            const p = r.probabilites[c] ?? 0
            return (
              <li key={c} className={css.ligne}>
                <span className={css.nom}>{t(`short.${c}`)}</span>
                <span className={css.piste} aria-hidden="true">
                  <span
                    className={css.remplissage}
                    style={{ width: `${p * 100}%`, background: `var(--c-${c})`, animationDelay: `${i * 120}ms` }}
                  />
                  <span className={css.repere} style={{ insetInlineStart: `${seuilPct}%` }} />
                </span>
                <span className={css.valeur} dir="ltr">{(p * 100).toFixed(1)} %</span>
              </li>
            )
          })}
        </ul>
      </section>
    </article>
  )
}
