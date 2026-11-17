/**
 * Fiches des maladies : textes volontairement prudents et généraux
 * (aucune dose, aucun produit). Toujours à faire confirmer par un ingénieur agronome.
 */
import type { ClasseMaladie } from '../types'
import type { Lang } from '../i18n'

export interface Fiche {
  description: string
  symptomes: string[]
  conseils: string[]
}

export const MALADIES: Record<ClasseMaladie, Record<Lang, Fiche>> = {
  healthy: {
    fr: {
      description:
        "Une feuille d'olivier saine est allongée, ferme, vert foncé dessus et gris argenté dessous, sans taches ni déformations.",
      symptomes: [
        'Couleur uniforme, sans taches circulaires ni décolorations',
        'Bord et limbe réguliers, non enroulés ni cloqués',
        'Revers argenté et lisse',
      ],
      conseils: [
        'Continuez à surveiller régulièrement vos arbres, surtout après les périodes humides.',
        'Un bon entretien (taille aérée, arbre peu encombré) favorise la santé du feuillage.',
        "Une seule feuille saine ne garantit pas la santé de tout l'arbre : observez plusieurs feuilles.",
      ],
    },
    ar: {
      description:
        'ورقة الزيتون السليمة طويلة وصلبة، خضراء داكنة من الأعلى ورمادية فضّية من الأسفل، بلا بقع ولا تشوّهات.',
      symptomes: [
        'لون متجانس بلا بقع دائرية أو شحوب',
        'حافة وسطح منتظمان، بلا التفاف أو انتفاخ',
        'الوجه السفلي فضّي وأملس',
      ],
      conseils: [
        'واصل مراقبة أشجارك بانتظام، خاصة بعد فترات الرطوبة.',
        'التقليم المهوّي وقلّة الازدحام داخل الشجرة يساعدان على سلامة الأوراق.',
        'ورقة سليمة واحدة لا تضمن سلامة الشجرة كلها: افحص عدة أوراق.',
      ],
    },
  },
  aculus: {
    fr: {
      description:
        "Aculus olearius est un acarien microscopique qui se nourrit sur les feuilles, les bourgeons et les jeunes fruits de l'olivier. Ses piqûres peuvent déformer et affaiblir les jeunes pousses.",
      symptomes: [
        'Feuilles ou jeunes pousses déformées, parfois enroulées ou cloquées',
        'Aspect terne, plombé ou bronzé du feuillage',
        'Croissance ralentie des jeunes rameaux, chute de feuilles dans les cas marqués',
        'Petits défauts ou déformations sur les jeunes fruits',
      ],
      conseils: [
        "Observez les jeunes pousses et les bourgeons, là où l'acarien est le plus actif.",
        "Évitez de déplacer du matériel végétal d'un verger touché vers un verger sain.",
        'Un arbre vigoureux, bien nourri et bien irrigué supporte mieux les attaques.',
        'Demandez conseil à un ingénieur agronome ou à votre service agricole local pour confirmer et choisir la conduite à tenir.',
      ],
    },
    ar: {
      description:
        'Aculus olearius حلم مجهري يتغذّى على أوراق الزيتون وبراعمه وثماره الصغيرة. قد تسبّب لدغاته تشوّه الأفرع الفتية وإضعافها.',
      symptomes: [
        'أوراق أو أفرع فتية مشوّهة، وقد تكون ملتفّة أو منتفخة',
        'مظهر باهت أو رصاصي أو برونزي للأوراق',
        'بطء نموّ الأغصان الصغيرة وتساقط الأوراق في الحالات الشديدة',
        'عيوب أو تشوّهات صغيرة على الثمار الصغيرة',
      ],
      conseils: [
        'راقب الأفرع الفتية والبراعم، حيث يكون الحلم أكثر نشاطًا.',
        'تجنّب نقل مواد نباتية من بستان مصاب إلى بستان سليم.',
        'الشجرة القوية جيدة التغذية والريّ تتحمّل الإصابة بشكل أفضل.',
        'استشر مهندسًا زراعيًا أو المصلحة الفلاحية المحلية للتأكّد من التشخيص واختيار الإجراء المناسب.',
      ],
    },
  },
  peacock_spot: {
    fr: {
      description:
        "L'œil de paon est une maladie fongique du feuillage, favorisée par l'humidité et la fraîcheur. Elle peut provoquer une chute importante des feuilles et affaiblir l'arbre.",
      symptomes: [
        "Taches rondes sombres, souvent entourées d'un halo jaunâtre (aspect d'« œil »)",
        'Taches visibles surtout sur la face supérieure des feuilles',
        'Jaunissement puis chute des feuilles atteintes',
        'Éclaircissement du feuillage dans les cas avancés',
      ],
      conseils: [
        "Favorisez l'aération de l'arbre (taille légère) pour limiter l'humidité dans la ramure.",
        'Ramassez et éloignez les feuilles tombées, qui peuvent conserver la maladie.',
        'Surveillez davantage après les périodes de pluie, de rosée ou de brouillard.',
        'Demandez conseil à un ingénieur agronome ou à votre service agricole local pour confirmer et choisir la conduite à tenir.',
      ],
    },
    ar: {
      description:
        'عين الطاووس مرض فطري يصيب الأوراق، تساعد عليه الرطوبة والبرودة. قد يسبّب تساقطًا كبيرًا للأوراق وإضعاف الشجرة.',
      symptomes: [
        'بقع دائرية داكنة، غالبًا يحيط بها هالة مصفرّة (شكل «العين»)',
        'البقع ظاهرة خاصة على الوجه العلوي للأوراق',
        'اصفرار ثم تساقط الأوراق المصابة',
        'خفّة الأوراق على الشجرة في الحالات المتقدّمة',
      ],
      conseils: [
        'ساعد على تهوية الشجرة (تقليم خفيف) للحدّ من الرطوبة داخل الأغصان.',
        'اجمع الأوراق المتساقطة وأبعدها، فقد تحمل المرض.',
        'زد من المراقبة بعد فترات المطر والندى والضباب.',
        'استشر مهندسًا زراعيًا أو المصلحة الفلاحية المحلية للتأكّد من التشخيص واختيار الإجراء المناسب.',
      ],
    },
  },
}
