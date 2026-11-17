/** État vide / chargement d'un panneau : garde la mise en page stable avant l'arrivée du résultat. */
import { useI18n } from '../i18n'
import LoadingLeaf from './LoadingLeaf'
import css from './EmptyState.module.css'

interface Props {
  type: 'diagnostic' | 'heatmap'
  enCours: boolean
}

export default function EmptyState({ type, enCours }: Props) {
  const { t } = useI18n()
  return (
    <div className={css.vide} role={enCours ? 'status' : undefined}>
      {enCours ? (
        <>
          <LoadingLeaf size={88} />
          <p>{t('analyzing')}</p>
        </>
      ) : (
        <>
          {type === 'diagnostic' ? (
            // Feuille de contour avec nervure
            <svg viewBox="0 0 96 96" width="88" height="88" aria-hidden="true" focusable="false">
              <path d="M14 82C16 40 44 14 84 12c0 40-24 68-70 70z" fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" strokeDasharray="2 7" strokeLinecap="round" />
              <path d="M14 82C34 58 56 36 78 18" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              <path d="M38 60l-2-14M52 46l-1-14M26 72l-12-4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" opacity="0.6" />
            </svg>
          ) : (
            // Carré de comparaison avec poignée et taches de chaleur
            <svg viewBox="0 0 96 96" width="88" height="88" aria-hidden="true" focusable="false">
              <rect x="10" y="10" width="76" height="76" rx="10" fill="none" stroke="currentColor" strokeWidth="3" strokeDasharray="2 7" strokeLinecap="round" />
              <circle cx="62" cy="38" r="12" fill="var(--terracotta)" opacity="0.28" />
              <circle cx="62" cy="38" r="5" fill="var(--terracotta)" opacity="0.45" />
              <path d="M48 10v76" stroke="currentColor" strokeWidth="3" />
              <circle cx="48" cy="48" r="9" fill="var(--white)" stroke="currentColor" strokeWidth="3" />
            </svg>
          )}
          <p>{t(type === 'diagnostic' ? 'emptyDiag' : 'emptyHeat')}</p>
        </>
      )}
    </div>
  )
}
