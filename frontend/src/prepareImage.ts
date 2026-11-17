/**
 * Reproduit côté navigateur le prétraitement du modèle (src/models/predict.py) :
 * image rendue carrée par un padding de la couleur du fond (médiane des bords haut/bas),
 * puis redimensionnée en 224×224. Sert de « avant » dans le comparateur.
 */
export const TAILLE_MAX_OCTETS = 10 * 1024 * 1024

export function verifierFichier(f: File): 'ok' | 'type' | 'taille' {
  if (!f.type.startsWith('image/')) return 'type'
  if (f.size > TAILLE_MAX_OCTETS) return 'taille'
  return 'ok'
}

export async function preparerImage(fichier: File, taille = 224): Promise<string> {
  const bmp = await createImageBitmap(fichier)

  // Réduction pour échantillonner les bords sans lire une image de 12 Mpx
  const echelle = Math.min(1, 400 / Math.max(bmp.width, bmp.height))
  const w = Math.max(1, Math.round(bmp.width * echelle))
  const h = Math.max(1, Math.round(bmp.height * echelle))
  const petit = document.createElement('canvas')
  petit.width = w
  petit.height = h
  const pc = petit.getContext('2d', { willReadFrequently: true })!
  pc.drawImage(bmp, 0, 0, w, h)

  const n = Math.min(3, h)
  const pixels = [...pc.getImageData(0, 0, w, n).data, ...pc.getImageData(0, h - n, w, n).data]
  const mediane = (c: number) => {
    const v: number[] = []
    for (let i = c; i < pixels.length; i += 4) v.push(pixels[i])
    v.sort((a, b) => a - b)
    return v[Math.floor(v.length / 2)] ?? 0
  }

  const sortie = document.createElement('canvas')
  sortie.width = sortie.height = taille
  const ctx = sortie.getContext('2d')!
  ctx.fillStyle = `rgb(${mediane(0)},${mediane(1)},${mediane(2)})`
  ctx.fillRect(0, 0, taille, taille)
  const k = taille / Math.max(bmp.width, bmp.height)
  const dw = bmp.width * k
  const dh = bmp.height * k
  ctx.drawImage(bmp, (taille - dw) / 2, (taille - dh) / 2, dw, dh)
  bmp.close()
  return sortie.toDataURL('image/jpeg', 0.92)
}
