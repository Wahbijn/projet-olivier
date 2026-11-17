/** Zone d'envoi : glisser-déposer, choix de fichier, appareil photo, aperçu et bouton d'analyse. */
import { useRef, useState, type DragEvent } from 'react'
import { useI18n } from '../i18n'
import LoadingLeaf from './LoadingLeaf'
import css from './UploadZone.module.css'

interface Props {
  apercu: string | null // URL de l'aperçu de la photo choisie
  enCours: boolean // analyse en cours
  erreur: string | null // message d'erreur déjà traduit
  onChoisir: (f: File) => void
  onAnalyser: () => void
}

export default function UploadZone({ apercu, enCours, erreur, onChoisir, onAnalyser }: Props) {
  const { t } = useI18n()
  const inputFichier = useRef<HTMLInputElement>(null)
  const inputCamera = useRef<HTMLInputElement>(null)
  const [survol, setSurvol] = useState(false)

  const prendre = (fichiers: FileList | null) => {
    const f = fichiers?.[0]
    if (f) onChoisir(f)
  }

  const deposer = (e: DragEvent) => {
    e.preventDefault()
    setSurvol(false)
    if (!enCours) prendre(e.dataTransfer.files)
  }

  return (
    <section className={css.carte} aria-labelledby="titre-envoi">
      <h2 id="titre-envoi" className="sr-only">
        {t('choose')}
      </h2>

      <div
        className={`${css.zone} ${survol ? css.survol : ''} ${apercu ? css.avecApercu : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setSurvol(true)
        }}
        onDragLeave={() => setSurvol(false)}
        onDrop={deposer}
      >
        {apercu ? (
          <div className={css.cadre}>
            <img src={apercu} alt={t('previewAlt')} className={css.apercu} />
            {enCours && (
              <div className={css.voile} aria-hidden="true">
                <LoadingLeaf />
              </div>
            )}
          </div>
        ) : (
          <div className={css.invite}>
            <svg viewBox="0 0 48 48" width="56" height="56" aria-hidden="true" focusable="false">
              <rect x="6" y="10" width="36" height="28" rx="6" fill="none" stroke="currentColor" strokeWidth="2.5" />
              <circle cx="24" cy="24" r="7" fill="none" stroke="currentColor" strokeWidth="2.5" />
              <path d="M17 10l2-4h10l2 4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
            </svg>
            <p className={css.titre}>{t('dropTitle')}</p>
            <p className={css.aide}>{t('dropHint')}</p>
          </div>
        )}
      </div>

      {/* Champs fichiers cachés, déclenchés par les boutons */}
      <input ref={inputFichier} type="file" accept="image/*" hidden onChange={(e) => { prendre(e.target.files); e.target.value = '' }} />
      <input ref={inputCamera} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { prendre(e.target.files); e.target.value = '' }} />

      <div className={css.actions}>
        <button type="button" className="btn" disabled={enCours} onClick={() => inputFichier.current?.click()}>
          {t('choose')}
        </button>
        <button type="button" className={`btn ${css.camera}`} disabled={enCours} onClick={() => inputCamera.current?.click()}>
          {t('takePhoto')}
        </button>
      </div>

      {erreur && (
        <p className={css.erreur} role="alert">
          {erreur}
        </p>
      )}

      {apercu && (
        <button type="button" className={`btn btn--primary ${css.analyser}`} disabled={enCours} aria-busy={enCours} onClick={onAnalyser}>
          {enCours ? t('analyzing') : t('analyze')}
        </button>
      )}
      {enCours && (
        <p className={css.attente} role="status">
          {t('analyzingHint')}
        </p>
      )}
    </section>
  )
}
