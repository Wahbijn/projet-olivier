/** Appels à l'API FastAPI. */
import type { Prediction, Sante } from './types'

export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '')

/** Type d'erreur → permet d'afficher un message humain traduit. */
export type ErreurType = 'invalide' | 'trop_lourde' | 'indisponible' | 'injoignable' | 'inconnue'

export class ApiError extends Error {
  constructor(public type: ErreurType, public detail?: string) {
    super(detail ?? type)
  }
}

const DELAI_MS = 90_000 // le calcul Grad-CAM peut être long sur CPU

async function appeler(url: string, init?: RequestInit): Promise<Response> {
  const ctrl = new AbortController()
  const minuteur = setTimeout(() => ctrl.abort(), DELAI_MS)
  try {
    return await fetch(url, { ...init, signal: ctrl.signal })
  } catch {
    // Réseau coupé, API éteinte, CORS ou délai dépassé
    throw new ApiError('injoignable')
  } finally {
    clearTimeout(minuteur)
  }
}

export async function lireSante(): Promise<Sante> {
  const rep = await appeler(`${API_URL}/health`)
  if (!rep.ok) throw new ApiError('injoignable')
  return rep.json()
}

export async function predire(fichier: Blob, nom: string): Promise<Prediction> {
  const form = new FormData()
  form.append('fichier', fichier, nom)
  const rep = await appeler(`${API_URL}/predict?heatmap=true`, { method: 'POST', body: form })
  if (rep.ok) return rep.json()

  let detail: string | undefined
  try {
    detail = (await rep.json()).detail
  } catch {
    /* corps non JSON : on ignore */
  }
  if (rep.status === 400) throw new ApiError('invalide', detail)
  if (rep.status === 413) throw new ApiError('trop_lourde', detail)
  if (rep.status === 503) throw new ApiError('indisponible', detail)
  throw new ApiError('inconnue', detail)
}
