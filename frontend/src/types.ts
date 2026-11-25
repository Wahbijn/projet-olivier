/** Types partagés avec le contrat de l'API FastAPI. */

export type ClasseMaladie = 'healthy' | 'aculus' | 'peacock_spot'
export type Classe = ClasseMaladie | 'incertain'

export interface Sante {
  statut: 'ok' | 'degrade'
  modele_charge: boolean
  architecture: string | null
  classes: string[]
  seuil_confiance: number | null
  erreur: string | null
}

export interface Prediction {
  classe: Classe
  classe_predite: ClasseMaladie
  confiance: number // 0 à 1
  incertain: boolean
  probabilites: Record<ClasseMaladie, number>
  seuil_confiance: number
  heatmap_base64: string | null
  /** Image réellement analysée (PNG 224×224), si l'API la renvoie */
  image_base64?: string | null
}

/** Une analyse conservée dans l'historique de la session (en mémoire uniquement). */
export interface Analyse {
  id: number
  date: Date
  /** Photo « préparée » (carrée, 224 px, comme celle vue par le modèle) */
  photoPreparee: string
  resultat: Prediction
}

export const CLASSES: ClasseMaladie[] = ['healthy', 'aculus', 'peacock_spot']

/** Contrat de POST /ask (assistant RAG). */
export interface RagSource {
  numero: number
  titre: string
  url: string | null
  extrait: string
  score: number
}

export interface RagReponse {
  reponse: string
  sources: RagSource[]
  refuse: boolean
}