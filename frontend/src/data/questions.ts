/** Questions proposées à l'agriculteur selon le diagnostic (un clic = une question envoyée à l'assistant RAG). */
import type { Lang } from '../i18n'
import type { Classe } from '../types'

export const QUESTIONS: Record<Classe, Record<Lang, string[]>> = {
  peacock_spot: {
    fr: [
      "Comment confirmer que c'est bien l'œil de paon ?",
      'Quand le risque est-il le plus élevé ?',
      'Comment limiter la propagation dans mon verger ?',
    ],
    ar: ['كيف أتأكد أنه مرض عين الطاووس؟', 'متى يكون الخطر أكبر؟', 'كيف أحدّ من انتشاره في بستاني؟'],
  },
  aculus: {
    fr: [
      "Comment confirmer la présence de l'acarien ?",
      'Quels dégâts peut-il causer ?',
      'Comment surveiller mes arbres ?',
    ],
    ar: ['كيف أتأكد من وجود الحلم؟', 'ما الأضرار التي قد يسببها؟', 'كيف أراقب أشجاري؟'],
  },
  healthy: {
    fr: [
      'Comment garder mes feuilles en bonne santé ?',
      'Quels signes surveiller après la pluie ?',
      'À quelle fréquence inspecter mes arbres ?',
    ],
    ar: ['كيف أحافظ على سلامة أوراق أشجاري؟', 'ما العلامات التي أراقبها بعد المطر؟', 'كم مرة أفحص أشجاري؟'],
  },
  incertain: {
    fr: [
      "Quels sont les symptômes de l'œil de paon ?",
      "Quels sont les symptômes de l'acarien Aculus olearius ?",
    ],
    ar: ['ما أعراض مرض عين الطاووس؟', 'ما أعراض حلم Aculus olearius؟'],
  },
}
