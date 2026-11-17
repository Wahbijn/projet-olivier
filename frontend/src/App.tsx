/** Page unique : accueil, envoi de la photo, résultat, historique, pied de page. */
import { useEffect, useRef, useState } from 'react'
import { ApiError, predire } from './api'
import ApiStatus from './components/ApiStatus'
import DiseaseInfo from './components/DiseaseInfo'
import EmptyState from './components/EmptyState'
import HeatmapCompare from './components/HeatmapCompare'
import History from './components/History'
import { OliveBranch, TilePattern } from './components/Ornaments'
import ResultCard from './components/ResultCard'
import UploadZone from './components/UploadZone'
import { useI18n, type Cle } from './i18n'
import { preparerImage, verifierFichier } from './prepareImage'
import type { Analyse } from './types'
import css from './App.module.css'

const MAX_HISTORIQUE = 12

export default function App() {
  const { t, lang, setLang } = useI18n()
  const [fichier, setFichier] = useState<File | null>(null)
  const [apercu, setApercu] = useState<string | null>(null)
  const [enCours, setEnCours] = useState(false)
  const [erreurCle, setErreurCle] = useState<Cle | null>(null)
  const [courante, setCourante] = useState<Analyse | null>(null)
  const [historique, setHistorique] = useState<Analyse[]>([])
  const compteur = useRef(0)
  const resultatRef = useRef<HTMLDivElement>(null)

  // Libère l'URL d'aperçu quand elle change ou à la fermeture
  useEffect(() => () => { if (apercu) URL.revokeObjectURL(apercu) }, [apercu])

  const choisir = (f: File) => {
    // Vérification côté client avant tout envoi
    const verdict = verifierFichier(f)
    if (verdict !== 'ok') {
      setErreurCle(verdict === 'type' ? 'errType' : 'errSize')
      return
    }
    setErreurCle(null)
    setFichier(f)
    setCourante(null)
    setApercu(URL.createObjectURL(f))
  }

  const analyser = async () => {
    if (!fichier || enCours) return
    setEnCours(true)
    setErreurCle(null)
    try {
      const resultat = await predire(fichier, fichier.name)
      // Image réellement analysée (renvoyée par l'API) ; à défaut, on la reconstitue côté client
      const photoPreparee = resultat.image_base64
        ? `data:image/png;base64,${resultat.image_base64}`
        : await preparerImage(fichier)
      const analyse: Analyse = { id: ++compteur.current, date: new Date(), photoPreparee, resultat }
      setCourante(analyse)
      setHistorique((h) => [analyse, ...h].slice(0, MAX_HISTORIQUE))
      // Amène le résultat à l'écran (utile sur smartphone)
      // (inutile sur ordinateur : tout est visible sans défiler)
      if (window.matchMedia('(max-width: 1023px)').matches) {
        requestAnimationFrame(() => resultatRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
      }
    } catch (e) {
      const type = e instanceof ApiError ? e.type : 'inconnue'
      setErreurCle(({ invalide: 'err400', trop_lourde: 'err413', indisponible: 'err503', injoignable: 'errDown', inconnue: 'errUnknown' } as const)[type])
    } finally {
      setEnCours(false)
    }
  }

  const nouvelle = () => {
    setFichier(null)
    setApercu(null)
    setCourante(null)
    setErreurCle(null)
    if (window.matchMedia('(max-width: 1023px)').matches) window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const classeFiche = courante ? courante.resultat.classe_predite : null

  return (
    <div className={css.app}>
      <TilePattern />
      <a href="#contenu" className={css.evitement}>{t('skip')}</a>

      <header className={css.entete}>
        <div className={css.marque}>
          <OliveBranch className={css.branche} />
          <h1 className={css.nom}>{t('appName')}</h1>
          <p className={css.slogan}>{t('tagline')}</p>
        </div>
        <div className={css.barre}>
          <ApiStatus />
          <button
            type="button"
            className={css.langue}
            onClick={() => setLang(lang === 'fr' ? 'ar' : 'fr')}
            aria-label={t('language')}
            lang={lang === 'fr' ? 'ar' : 'fr'}
          >
            {lang === 'fr' ? 'العربية' : 'Français'}
          </button>
        </div>
      </header>

      <main id="contenu" className={css.principal}>
        {/* ===== Colonne gauche : envoi, conseils, historique ===== */}
        <div className={css.colGauche}>
          {(fichier || courante) && (
            <button type="button" className={`btn ${css.nouvelle}`} onClick={nouvelle} disabled={enCours}>
              {t('newAnalysis')}
            </button>
          )}

          <div className={css.envoi}>
            <UploadZone
              apercu={apercu}
              enCours={enCours}
              erreur={erreurCle ? t(erreurCle) : null}
              onChoisir={choisir}
              onAnalyser={analyser}
            />
          </div>

          <section className={css.conseils} aria-labelledby="titre-conseils">
            <h2 id="titre-conseils">{t('tipsTitle')}</h2>
            <ol>
              <li><span>1</span>{t('tip1')}</li>
              <li><span>2</span>{t('tip2')}</li>
              <li><span>3</span>{t('tip3')}</li>
            </ol>
          </section>

          <div className={css.historique}>
            <History analyses={historique} activeId={courante?.id ?? null} onSelect={setCourante} />
          </div>
        </div>

        {/* ===== Colonne centrale : diagnostic + fiche en onglets ===== */}
        <div className={css.colCentre}>
          {courante && !enCours ? (
            <>
              <div ref={resultatRef} className={css.resultat}>
                <ResultCard analyse={courante} />
              </div>
              <div className={css.fiche}>
                <DiseaseInfo key={classeFiche} classe={classeFiche!} />
              </div>
            </>
          ) : (
            <div className={css.etatVide}>
              <EmptyState type="diagnostic" enCours={enCours} />
            </div>
          )}
        </div>

        {/* ===== Colonne droite : comparateur Grad-CAM en grand ===== */}
        <section
          className={`${css.colDroite} ${courante && !enCours ? '' : css.etatVide}`}
          aria-labelledby="titre-heatmap"
        >
          <h2 id="titre-heatmap" className={css.titrePanneau}>{t('heatmapTitle')}</h2>
          <div className={css.corpsPanneau}>
            {courante && !enCours ? (
              <HeatmapCompare key={courante.id} avant={courante.photoPreparee} heatmapBase64={courante.resultat.heatmap_base64} />
            ) : (
              <EmptyState type="heatmap" enCours={enCours} />
            )}
          </div>
        </section>
      </main>

      <footer className={css.pied}>
        <p className={css.avertissement}>{t('footerWarning')}</p>
        <p>{t('footerModel')}</p>
      </footer>
    </div>
  )
}
