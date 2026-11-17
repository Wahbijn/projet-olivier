/** Petite feuille qui « pousse » : animation de chargement partagée. */
import css from './LoadingLeaf.module.css'

export default function LoadingLeaf({ size = 96 }: { size?: number }) {
  return (
    <svg className={css.pousse} width={size} height={size} viewBox="0 0 80 80" aria-hidden="true" focusable="false">
      <path className={css.tige} d="M40 74 C40 56 40 40 40 22" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" pathLength="1" />
      <path className={`${css.feuille} ${css.f1}`} d="M40 58 C28 56 20 48 18 38 C30 38 38 46 40 58Z" fill="currentColor" />
      <path className={`${css.feuille} ${css.f2}`} d="M40 46 C52 44 60 36 62 26 C50 26 42 34 40 46Z" fill="currentColor" />
      <path className={`${css.feuille} ${css.f3}`} d="M40 22 C32 16 32 8 40 2 C48 8 48 16 40 22Z" fill="currentColor" />
    </svg>
  )
}
