/** Traductions FR / AR (l'arabe passe la page en RTL) et petit contexte React. */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

export type Lang = 'fr' | 'ar'

const fr = {
  appName: 'Zitouna',
  tagline: 'La santé de votre olivier, feuille par feuille.',
  intro:
    "Prenez une feuille en photo : l'intelligence artificielle repère l'acarien Aculus olearius ou l'œil de paon, et vous montre où elle a regardé.",
  tipsTitle: 'Pour une bonne photo',
  tip1: 'Une seule feuille, bien au centre',
  tip2: 'Bien éclairée, sans ombre dure ni reflet',
  tip3: 'Sur un fond uni si possible (la main, une feuille de papier)',
  dropTitle: 'Déposez une photo ici',
  dropHint: 'ou choisissez-la depuis votre appareil · JPG, PNG · 10 Mo max',
  choose: 'Choisir une photo',
  takePhoto: 'Prendre une photo',
  analyze: 'Analyser la feuille',
  analyzing: 'Analyse en cours…',
  analyzingHint: 'Le modèle observe la feuille. Cela peut prendre quelques secondes.',
  previewAlt: 'Aperçu de la photo choisie',
  errType: "Ce fichier n'est pas une image. Choisissez une photo (JPG, PNG…).",
  errSize: 'Cette photo est trop lourde (10 Mo maximum). Réduisez sa taille ou reprenez-la.',
  err400: "L'analyse n'a pas pu lire ce fichier comme une image. Essayez une autre photo.",
  err413: 'La photo est trop lourde pour le serveur (10 Mo maximum).',
  err503: "Le modèle n'est pas encore chargé sur le serveur. Patientez un instant puis réessayez.",
  errDown: "Impossible de joindre le serveur d'analyse. Vérifiez votre connexion et que l'API est bien lancée.",
  errUnknown: "Une erreur inattendue s'est produite. Réessayez dans un instant.",
  apiOk: 'Serveur prêt',
  apiDegraded: 'Serveur actif, modèle indisponible',
  apiDown: 'Serveur injoignable',
  apiChecking: 'Vérification…',
  resultTitle: 'Résultat de l’analyse',
  confidence: 'Confiance',
  thresholdLabel: 'seuil {s} %',
  gaugeLabel: 'Confiance du modèle : {c} %, seuil à {s} %',
  probabilities: 'Probabilités par classe',
  uncertainTitle: 'Je n’en suis pas sûr',
  uncertainText:
    "Ce n'est pas une erreur : le modèle n'est pas assez confiant pour trancher. À titre indicatif, la feuille ressemble le plus à :",
  uncertainAdviceTitle: 'Pour une meilleure photo',
  uncertainAdvice1: 'Rapprochez-vous : la feuille doit remplir la majeure partie de l’image.',
  uncertainAdvice2: 'Placez-vous à l’ombre légère ou face à la lumière, sans reflet.',
  uncertainAdvice3: 'Posez la feuille sur un fond uni, puis reprenez la photo bien nette.',
  heatmapTitle: 'Où le modèle a regardé',
  heatmapLegend: 'Les zones rouges sont celles qui ont le plus guidé le modèle.',
  heatmapSlider: 'Curseur de comparaison entre la photo et la heatmap',
  before: 'Photo',
  after: 'Heatmap',
  heatmapAltBefore: 'Photo de la feuille préparée pour le modèle',
  heatmapAltAfter: 'Feuille avec la heatmap Grad-CAM superposée',
  noHeatmap: 'La heatmap n’est pas disponible pour cette analyse.',
  indicative: 'Informations indicatives — à confirmer avec un ingénieur agronome.',
  infoTitle: 'Fiche : {name}',
  description: 'Description',
  symptoms: 'Symptômes à observer',
  advice: 'Conseils généraux',
  historyTitle: 'Historique de la session',
  historyEmpty: 'Vos dernières analyses apparaîtront ici.',
  historyItem: 'Revoir l’analyse n°{n} : {name}',
  historyNote: 'Conservé en mémoire seulement : effacé à la fermeture de la page.',
  newAnalysis: 'Nouvelle analyse',
  footerWarning: 'Prototype pédagogique d’aide au diagnostic, ne remplace pas l’avis d’un expert.',
  footerModel: 'Modèle : ResNet50 · seuil de confiance 0,8',
  language: 'Langue',
  skip: 'Aller au contenu',
  'class.healthy': 'Feuille saine',
  'class.aculus': 'Aculus olearius (acarien)',
  'class.peacock_spot': 'Œil de paon',
  'class.incertain': 'Diagnostic incertain',
  'short.healthy': 'Saine',
  'short.aculus': 'Aculus',
  'short.peacock_spot': 'Œil de paon',
  tabDesc: 'Description',
  tabSymptoms: 'Symptômes',
  tabAdvice: 'Conseils',
  emptyDiag: 'Le diagnostic apparaîtra ici',
  emptyHeat: 'La carte de chaleur apparaîtra ici',
  tabsLabel: 'Fiche maladie',
    ragTitle: 'Poser une question',
  ragIntro: 'Réponses tirées de documents agronomiques, avec leurs sources.',
  ragSuggestions: 'Questions fréquentes',
  ragLabel: 'Votre question sur cette feuille',
  ragPlaceholder: 'Votre question…',
  ragSend: 'Envoyer',
  ragThinking: 'Recherche dans les documents…',
  ragSources: 'Sources ({n})',
  ragOpenSource: 'Ouvrir la source',
  ragYou: 'Vous',
  ragBot: 'Assistant',
  ragDisclaimer: 'Réponses générées par IA à partir de documents : à confirmer avec un ingénieur agronome.',
  ragUnavailable: "L'assistant n'est pas disponible pour le moment.",
  ragErrDown: 'Impossible de joindre le serveur. Réessayez dans un instant.',
  ragErrGeneric: "L'assistant n'a pas pu répondre. Réessayez dans un instant.",
}

export type Cle = keyof typeof fr

const ar: Record<Cle, string> = {
  appName: 'زيتونة',
  tagline: 'صحّة زيتونتك، ورقةً ورقة.',
  intro:
    'صوّر ورقة الزيتون: يكشف الذكاء الاصطناعي حلم Aculus olearius أو مرض عين الطاووس، ويُريك المناطق التي ركّز عليها.',
  tipsTitle: 'للحصول على صورة جيدة',
  tip1: 'ورقة واحدة فقط، في وسط الصورة',
  tip2: 'إضاءة جيدة بلا ظلّ قاسٍ ولا انعكاس',
  tip3: 'على خلفية موحّدة إن أمكن (اليد، ورقة بيضاء)',
  dropTitle: 'أفلت الصورة هنا',
  dropHint: 'أو اخترها من جهازك · JPG، PNG · الحدّ الأقصى 10 ميغابايت',
  choose: 'اختيار صورة',
  takePhoto: 'التقاط صورة',
  analyze: 'تحليل الورقة',
  analyzing: 'جارٍ التحليل…',
  analyzingHint: 'النموذج يفحص الورقة. قد يستغرق ذلك بضع ثوانٍ.',
  previewAlt: 'معاينة الصورة المختارة',
  errType: 'هذا الملف ليس صورة. اختر صورة (JPG، PNG…).',
  errSize: 'الصورة ثقيلة جدًا (الحدّ الأقصى 10 ميغابايت). صغّر حجمها أو أعد التقاطها.',
  err400: 'تعذّرت قراءة هذا الملف كصورة. جرّب صورة أخرى.',
  err413: 'الصورة ثقيلة جدًا بالنسبة للخادم (الحدّ الأقصى 10 ميغابايت).',
  err503: 'النموذج لم يُحمَّل بعد على الخادم. انتظر لحظة ثم أعد المحاولة.',
  errDown: 'تعذّر الاتصال بخادم التحليل. تحقّق من اتصالك ومن تشغيل الـ API.',
  errUnknown: 'حدث خطأ غير متوقّع. أعد المحاولة بعد قليل.',
  apiOk: 'الخادم جاهز',
  apiDegraded: 'الخادم يعمل لكنّ النموذج غير متاح',
  apiDown: 'تعذّر الوصول إلى الخادم',
  apiChecking: 'جارٍ التحقّق…',
  resultTitle: 'نتيجة التحليل',
  confidence: 'الثقة',
  thresholdLabel: 'العتبة {s}٪',
  gaugeLabel: 'ثقة النموذج: {c}٪، العتبة {s}٪',
  probabilities: 'الاحتمالات حسب الفئة',
  uncertainTitle: 'لستُ متأكدًا',
  uncertainText: 'هذا ليس خطأً: النموذج غير واثق بما يكفي للحسم. على سبيل الإرشاد، تشبه الورقة أكثر:',
  uncertainAdviceTitle: 'لصورة أفضل',
  uncertainAdvice1: 'اقترب أكثر: يجب أن تملأ الورقة معظم الصورة.',
  uncertainAdvice2: 'صوّر في ظلّ خفيف أو في اتجاه الضوء، بلا انعكاسات.',
  uncertainAdvice3: 'ضع الورقة على خلفية موحّدة ثم أعد التصوير بوضوح.',
  heatmapTitle: 'أين نظر النموذج',
  heatmapLegend: 'المناطق الحمراء هي التي وجّهت النموذج أكثر من غيرها.',
  heatmapSlider: 'شريط المقارنة بين الصورة والخريطة الحرارية',
  before: 'الصورة',
  after: 'الخريطة الحرارية',
  heatmapAltBefore: 'صورة الورقة بعد تهيئتها للنموذج',
  heatmapAltAfter: 'الورقة مع الخريطة الحرارية Grad-CAM',
  noHeatmap: 'الخريطة الحرارية غير متوفرة لهذا التحليل.',
  indicative: 'معلومات إرشادية — يجب تأكيدها مع مهندس زراعي.',
  infoTitle: 'بطاقة: {name}',
  description: 'الوصف',
  symptoms: 'الأعراض الواجب ملاحظتها',
  advice: 'نصائح عامة',
  historyTitle: 'سجلّ الجلسة',
  historyEmpty: 'ستظهر هنا آخر تحليلاتك.',
  historyItem: 'عرض التحليل رقم {n}: {name}',
  historyNote: 'محفوظ في الذاكرة فقط: يُمحى عند إغلاق الصفحة.',
  newAnalysis: 'تحليل جديد',
  footerWarning: 'نموذج تعليمي للمساعدة على التشخيص، لا يغني عن رأي الخبير.',
  footerModel: 'النموذج: ResNet50 · عتبة الثقة 0٫8',
  language: 'اللغة',
  skip: 'انتقل إلى المحتوى',
  'class.healthy': 'ورقة سليمة',
  'class.aculus': 'حلم الزيتون (Aculus olearius)',
  'class.peacock_spot': 'عين الطاووس',
  'class.incertain': 'تشخيص غير مؤكّد',
  'short.healthy': 'سليمة',
  'short.aculus': 'الحلم',
  'short.peacock_spot': 'عين الطاووس',
  tabDesc: 'الوصف',
  tabSymptoms: 'الأعراض',
  tabAdvice: 'نصائح',
  emptyDiag: 'سيظهر التشخيص هنا',
  emptyHeat: 'ستظهر الخريطة الحرارية هنا',
  tabsLabel: 'بطاقة المرض',
    ragTitle: 'اطرح سؤالًا',
  ragIntro: 'إجابات مستخرجة من وثائق زراعية، مع ذكر مصادرها.',
  ragSuggestions: 'أسئلة شائعة',
  ragLabel: 'سؤالك حول هذه الورقة',
  ragPlaceholder: 'سؤالك…',
  ragSend: 'إرسال',
  ragThinking: 'جارٍ البحث في الوثائق…',
  ragSources: 'المصادر ({n})',
  ragOpenSource: 'فتح المصدر',
  ragYou: 'أنت',
  ragBot: 'المساعد',
  ragDisclaimer: 'إجابات مولّدة بالذكاء الاصطناعي من وثائق: يجب تأكيدها مع مهندس زراعي.',
  ragUnavailable: 'المساعد غير متاح حاليًا.',
  ragErrDown: 'تعذّر الاتصال بالخادم. أعد المحاولة بعد قليل.',
  ragErrGeneric: 'تعذّر على المساعد الإجابة. أعد المحاولة بعد قليل.',
}

const dicos: Record<Lang, Record<Cle, string>> = { fr, ar }

interface Ctx {
  lang: Lang
  setLang: (l: Lang) => void
  t: (cle: Cle, vars?: Record<string, string | number>) => string
}

const I18nContext = createContext<Ctx>(null!)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    try {
      return localStorage.getItem('zitouna-lang') === 'ar' ? 'ar' : 'fr'
    } catch {
      return 'fr'
    }
  })

  // Direction et langue du document (RTL pour l'arabe)
  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr'
    document.title = lang === 'ar' ? 'زيتونة – صحّة الزيتون' : "Zitouna – Santé de l'olivier"
    try {
      localStorage.setItem('zitouna-lang', lang)
    } catch {
      /* stockage indisponible : sans importance */
    }
  }, [lang])

  const t = useCallback<Ctx['t']>(
    (cle, vars) => {
      let s = dicos[lang][cle]
      if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v))
      return s
    },
    [lang],
  )

  const valeur = useMemo(() => ({ lang, setLang, t }), [lang, t])
  return <I18nContext.Provider value={valeur}>{children}</I18nContext.Provider>
}

export const useI18n = () => useContext(I18nContext)
