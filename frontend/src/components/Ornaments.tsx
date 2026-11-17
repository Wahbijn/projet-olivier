/** Motifs SVG originaux, dessinés à la main : branche d'olivier, carreaux, icônes de classe. */
import type { Classe } from '../types'

/** Branche d'olivier : tige courbe, feuilles lancéolées et quelques olives. */
export function OliveBranch({ className }: { className?: string }) {
  // Positions des feuilles le long de la tige : [x, y, angle, échelle]
  const feuilles: [number, number, number, number][] = [
    [40, 92, -150, 1],
    [58, 80, 20, 1],
    [82, 70, -155, 1.05],
    [104, 60, 15, 1.05],
    [128, 52, -160, 1],
    [150, 46, 10, 1],
    [172, 42, -165, 0.9],
    [196, 40, 0, 0.85],
  ]
  return (
    <svg className={className} viewBox="0 0 230 120" aria-hidden="true" focusable="false">
      <path d="M8 104 C60 100 120 70 214 38" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      {feuilles.map(([x, y, a, s], i) => (
        <path
          key={i}
          d="M0 0 C 8 -6 26 -7 40 0 C 26 7 8 6 0 0 Z"
          transform={`translate(${x} ${y}) rotate(${a}) scale(${s})`}
          fill="currentColor"
          opacity={i % 2 ? 0.55 : 0.85}
        />
      ))}
      <ellipse cx="94" cy="84" rx="5" ry="6.5" fill="var(--terracotta)" opacity="0.85" transform="rotate(-20 94 84)" />
      <ellipse cx="138" cy="70" rx="4.5" ry="6" fill="var(--terracotta)" opacity="0.85" transform="rotate(-25 138 70)" />
    </svg>
  )
}

/** Motif de carreaux : étoile à huit branches inspirée de la géométrie des zelliges. Très discret. */
export function TilePattern() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      style={{ position: 'fixed', inset: 0, width: '100%', height: '100%', zIndex: -1, pointerEvents: 'none' }}
    >
      <defs>
        <pattern id="carreau" width="72" height="72" patternUnits="userSpaceOnUse">
          <g fill="none" stroke="var(--olive-700)" strokeWidth="1.2" opacity="0.07">
            <rect x="21" y="21" width="30" height="30" />
            <rect x="21" y="21" width="30" height="30" transform="rotate(45 36 36)" />
            <circle cx="36" cy="36" r="6" />
            <path d="M0 0l12 12M72 0L60 12M0 72l12-12M72 72L60 60" />
          </g>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#carreau)" />
    </svg>
  )
}

/** Petite icône par classe (feuille saine, acarien, œil de paon). */
export function ClassIcon({ classe, size = 28 }: { classe: Classe; size?: number }) {
  const p = { width: size, height: size, viewBox: '0 0 32 32', 'aria-hidden': true, focusable: false } as const
  switch (classe) {
    case 'healthy': // feuille avec nervure
      return (
        <svg {...p}>
          <path d="M5 27C6 12 16 5 28 5c0 13-8 22-23 22z" fill="currentColor" opacity="0.9" />
          <path d="M5 27C12 19 19 13 25 8" stroke="var(--white)" strokeWidth="2" fill="none" strokeLinecap="round" />
        </svg>
      )
    case 'aculus': // acarien stylisé : corps allongé, pattes
      return (
        <svg {...p}>
          <ellipse cx="16" cy="16" rx="5" ry="10" fill="currentColor" />
          <g stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M11 11L4 8M11 16H3M11 21L4 24M21 11l7-3M21 16h8M21 21l7 3" />
          </g>
        </svg>
      )
    case 'peacock_spot': // tache en « œil » : anneaux concentriques
      return (
        <svg {...p}>
          <circle cx="16" cy="16" r="12" fill="currentColor" opacity="0.25" />
          <circle cx="16" cy="16" r="8" fill="none" stroke="currentColor" strokeWidth="2.4" />
          <circle cx="16" cy="16" r="3.4" fill="currentColor" />
        </svg>
      )
    default: // incertain : point d'interrogation
      return (
        <svg {...p}>
          <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" strokeWidth="2.4" />
          <path d="M11.5 12.5a4.5 4.5 0 1 1 6.4 4c-1.4.8-1.9 1.5-1.9 3" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
          <circle cx="16" cy="23.5" r="1.6" fill="currentColor" />
        </svg>
      )
  }
}
