/** Jauge de confiance en demi-cercle ; le seuil est marqué par un trait. Se remplit à l'apparition. */
import { useEffect, useState } from 'react'
import { useI18n } from '../i18n'
import css from './ConfidenceGauge.module.css'

interface Props {
  confiance: number // 0 à 1
  seuil: number // 0 à 1
  couleur: string // couleur CSS de la classe
}

const R = 80
const CX = 100
const CY = 100

export default function ConfidenceGauge({ confiance, seuil, couleur }: Props) {
  const { t } = useI18n()
  const [valeur, setValeur] = useState(0)

  // On passe de 0 à la vraie valeur après le montage pour déclencher la transition CSS
  useEffect(() => {
    const id = requestAnimationFrame(() => setValeur(confiance))
    return () => cancelAnimationFrame(id)
  }, [confiance])

  const pct = Math.round(confiance * 100)
  const seuilPct = Math.round(seuil * 100)

  // Position du trait de seuil sur l'arc (angle π·s mesuré depuis la gauche)
  const a = Math.PI * seuil
  const x1 = CX - (R - 12) * Math.cos(a)
  const y1 = CY - (R - 12) * Math.sin(a)
  const x2 = CX - (R + 12) * Math.cos(a)
  const y2 = CY - (R + 12) * Math.sin(a)
  const xl = CX - (R + 24) * Math.cos(a)
  const yl = CY - (R + 24) * Math.sin(a)

  return (
    <figure className={css.jauge} role="img" aria-label={t('gaugeLabel', { c: pct, s: seuilPct })}>
      <svg viewBox="-20 0 240 128" aria-hidden="true" focusable="false">
        <path d={`M${CX - R} ${CY} A${R} ${R} 0 0 1 ${CX + R} ${CY}`} className={css.fond} pathLength="100" />
        <path
          d={`M${CX - R} ${CY} A${R} ${R} 0 0 1 ${CX + R} ${CY}`}
          className={css.valeur}
          style={{ stroke: couleur, strokeDashoffset: 100 - valeur * 100 }}
          pathLength="100"
        />
        <line x1={x1} y1={y1} x2={x2} y2={y2} className={css.seuil} />
      </svg>
      {/* Étiquette du seuil en HTML : texte lisible (≥ 14 px) quelle que soit la taille de la jauge */}
      <span className={css.seuilTexte} style={{ left: `${((xl + 20) / 240) * 100}%`, top: `${(yl / 128) * 100}%` }}>
        {t('thresholdLabel', { s: seuilPct })}
      </span>
      <figcaption className={css.legende}>
        <span className={css.pct} dir="ltr" style={{ color: couleur }}>
          {pct} %
        </span>
        <span>{t('confidence')}</span>
      </figcaption>
    </figure>
  )
}
