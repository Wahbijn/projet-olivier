/** Comparateur avant/après : un curseur glissant révèle la photo préparée ou la heatmap Grad-CAM. */
import { useState } from 'react'
import { useI18n } from '../i18n'
import css from './HeatmapCompare.module.css'

interface Props {
  avant: string // photo préparée (URL de données)
  heatmapBase64: string | null
}

export default function HeatmapCompare({ avant, heatmapBase64 }: Props) {
  const { t } = useI18n()
  const [pos, setPos] = useState(50) // % de la largeur occupé par la photo

  if (!heatmapBase64) return <p className={css.absente}>{t('noHeatmap')}</p>
  const apres = `data:image/png;base64,${heatmapBase64}`

  return (
    <div className={css.bloc}>
      {/* dir=ltr : le curseur reste de gauche à droite, même en arabe */}
      <div className={css.zone}>
      <div className={css.comparateur} dir="ltr">
        <img className={css.img} src={apres} alt={t('heatmapAltAfter')} />
        <img
          className={`${css.img} ${css.dessus}`}
          src={avant}
          alt={t('heatmapAltBefore')}
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        />
        <span className={`${css.etiquette} ${css.gauche}`} aria-hidden="true">{t('before')}</span>
        <span className={`${css.etiquette} ${css.droite}`} aria-hidden="true">{t('after')}</span>
        <div className={css.poignee} style={{ left: `${pos}%` }} aria-hidden="true">
          <span className={css.bouton}>⟷</span>
        </div>
        {/* Vrai champ « range » transparent : clavier, tactile et lecteurs d'écran gratuits */}
        <input
          type="range"
          min={0}
          max={100}
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          className={css.range}
          aria-label={t('heatmapSlider')}
        />
      </div>
      </div>
      <p className={css.legende}>
        <span className={css.degrade} aria-hidden="true" />
        {t('heatmapLegend')}
      </p>
    </div>
  )
}
