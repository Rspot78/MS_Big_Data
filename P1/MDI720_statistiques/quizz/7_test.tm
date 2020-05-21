<TeXmacs|1.99.11>

<style|<tuple|article|std-latex>>

<\body>
  <\hide-preamble>
    <assign|Gread|<macro|1|<IfFileExists|"<Gin@base>".bb><Gread@eps|<Gin@base>.bb><Gread@@xetex@aux><arg|1>>>

    <assign|tightlist|<macro|>>

    <assign|Shaded|<\macro|body>
      <arg|body>
    </macro>>

    <assign|KeywordTok|<macro|1|<with|color|rgb:0.00,0.44,0.13|font-series|bold|<arg|1>>>>

    <assign|DataTypeTok|<macro|1|<with|color|rgb:0.56,0.13,0.00|<arg|1>>>>

    <assign|DecValTok|<macro|1|<with|color|rgb:0.25,0.63,0.44|<arg|1>>>>

    <assign|BaseNTok|<macro|1|<with|color|rgb:0.25,0.63,0.44|<arg|1>>>>

    <assign|FloatTok|<macro|1|<with|color|rgb:0.25,0.63,0.44|<arg|1>>>>

    <assign|CharTok|<macro|1|<with|color|rgb:0.25,0.44,0.63|<arg|1>>>>

    <assign|StringTok|<macro|1|<with|color|rgb:0.25,0.44,0.63|<arg|1>>>>

    <assign|CommentTok|<macro|1|<with|color|rgb:0.38,0.63,0.69|font-shape|italic|<arg|1>>>>

    <assign|OtherTok|<macro|1|<with|color|rgb:0.00,0.44,0.13|<arg|1>>>>

    <assign|AlertTok|<macro|1|<with|color|rgb:1.00,0.00,0.00|font-series|bold|<arg|1>>>>

    <assign|FunctionTok|<macro|1|<with|color|rgb:0.02,0.16,0.49|<arg|1>>>>

    <assign|RegionMarkerTok|<macro|1|<arg|1>>>

    <assign|ErrorTok|<macro|1|<with|color|rgb:1.00,0.00,0.00|font-series|bold|<arg|1>>>>

    <assign|NormalTok|<macro|1|<arg|1>>>

    <assign|ConstantTok|<macro|1|<with|color|rgb:0.53,0.00,0.00|<arg|1>>>>

    <assign|SpecialCharTok|<macro|1|<with|color|rgb:0.25,0.44,0.63|<arg|1>>>>

    <assign|VerbatimStringTok|<macro|1|<with|color|rgb:0.25,0.44,0.63|<arg|1>>>>

    <assign|SpecialStringTok|<macro|1|<with|color|rgb:0.73,0.40,0.53|<arg|1>>>>

    <assign|ImportTok|<macro|1|<arg|1>>>

    <assign|DocumentationTok|<macro|1|<with|color|rgb:0.73,0.13,0.13|font-shape|italic|<arg|1>>>>

    <assign|AnnotationTok|<macro|1|<with|color|rgb:0.38,0.63,0.69|font-series|bold|font-shape|italic|<arg|1>>>>

    <assign|CommentVarTok|<macro|1|<with|color|rgb:0.38,0.63,0.69|font-series|bold|font-shape|italic|<arg|1>>>>

    <assign|VariableTok|<macro|1|<with|color|rgb:0.10,0.09,0.49|<arg|1>>>>

    <assign|ControlFlowTok|<macro|1|<with|color|rgb:0.00,0.44,0.13|font-series|bold|<arg|1>>>>

    <assign|OperatorTok|<macro|1|<with|color|rgb:0.40,0.40,0.40|<arg|1>>>>

    <assign|BuiltInTok|<macro|1|<arg|1>>>

    <assign|ExtensionTok|<macro|1|<arg|1>>>

    <assign|PreprocessorTok|<macro|1|<with|color|rgb:0.74,0.48,0.00|<arg|1>>>>

    <assign|AttributeTok|<macro|1|<with|color|rgb:0.49,0.56,0.16|<arg|1>>>>

    <assign|InformationTok|<macro|1|<with|color|rgb:0.38,0.63,0.69|font-series|bold|font-shape|italic|<arg|1>>>>

    <assign|WarningTok|<macro|1|<with|color|rgb:0.38,0.63,0.69|font-series|bold|font-shape|italic|<arg|1>>>>

    <assign|br|<macro|<space|fill><next-line>* >>

    <assign|gt|<macro|\<gtr\>>>

    <assign|lt|<macro|\<less\>>>

    <assign|TeX|<macro|<with|font-family|rm|<Oldtex>>>>

    <assign|LaTeX|<macro|<with|font-family|rm|<Oldlatex>>>>

    <assign|PY|<macro|<let><PY@it>=<let><PY@bf>=<let><PY@ul>=<let><PY@tc>=<let><PY@bc>=<let><PY@ff>=>>

    <assign|PY|<macro|1|<csname>PY@tok@<arg|1><endcsname>>>

    <assign|PY|<macro|1|<ifx><arg|1><empty><else><PY@tok|<arg|1>><expandafter><PY@toks><fi>>>

    <assign|PY|<macro|1|<PY@bc|<PY@tc|<PY@ul|<PY@it|<PY@bf|<PY@ff|<arg|1>>>>>>>>>

    <assign|PY|<macro|1|2|<PY@reset><PY@toks><arg|1>++<PY@do|<arg|2>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.73,0.73|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.74,0.48,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.50,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.69,0.00,0.25|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.67,0.13,1.00|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.50,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.00,1.00|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.00,1.00|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.00,1.00|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.82,0.25,0.23|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.53,0.00,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.63,0.63,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.60,0.60,0.60|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.49,0.56,0.16|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.67,0.13,1.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.73,0.13,0.13|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.73,0.40,0.53|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.73,0.40,0.13|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.40,0.53|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.50,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.00,0.50|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.50,0.00,0.50|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.63,0.00,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.63,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:1.00,0.00,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@it>=<textit>>>

    <assign|csname|<macro|<let><PY@bf>=<textbf>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.00,0.50|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.53,0.53,0.53|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.27,0.87|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|>><fcolorbox|[|r|g>b]1.00,0.00,0.001,1,1<resize||0pt|-0.3bls|0pt|0.7bls>#<1>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<let><PY@bf>=<with|font-series|bold|<def>><PY@tc>#<1><with|color|rgb:0.00,0.50,0.00|#<1>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.50,0.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.00,0.00,1.00|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.10,0.09,0.49|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.73,0.13,0.13|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<assign|PY|<macro|1|<with|color|rgb:0.40,0.40,0.40|#<arg|1>>>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|csname|<macro|<let><PY@it>=<with|font-shape|italic|<def>><PY@tc>#<1><with|color|rgb:0.25,0.50,0.50|#<1>>>>

    <assign|PYZbs|<macro|`<next-line>>>

    <assign|PYZus|<macro|`_>>

    <assign|PYZob|<macro|`{>>

    <assign|PYZcb|<macro|`}>>

    <assign|PYZca|<macro|`<^>>>

    <assign|PYZam|<macro|`&>>

    <assign|PYZlt|<macro|`<\<>>>

    <assign|PYZgt|<macro|`<\>>>>

    <assign|PYZsh|<macro|`#>>

    <assign|PYZpc|<macro|`%>>

    <assign|PYZdl|<macro|`$>>

    <assign|PYZhy|<macro|`>>

    <assign|PYZsq|<macro|`<'>>>

    <assign|PYZdq|<macro|`<">>>

    <assign|PYZti|<macro|`<~>>>

    <assign|PYZat|<macro|@>>

    <assign|PYZlb|<macro|[>>

    <assign|PYZrb|<macro|]>>

    <assign|*|<macro|<Wrappedvisiblespace>>>

    <assign|*|<macro|<Wrappedcontinuationsymbol>>>

    <assign|*|<macro|<Wrappedcontinuationindent>>>

    <assign|*|<macro|<Wrappedafterbreak>>>

    <assign|*|<macro|<Wrappedbreaksatspecials>>>

    <assign|*|<macro|<Wrappedbreaksatpunct>>>

    <assign|Verbatim*|<\macro|1>
      <sbox><Wrappedcontinuationbox|<Wrappedcontinuationsymbol>><sbox><Wrappedvisiblespacebox|<FV@SetupFont><Wrappedvisiblespace>><assign|FancyVerbFormatLine|<macro|1|<hsize>tex-line-width
      <vtop|<with|par-mode|left|<hyphenpenalty>0pt<exhyphenpenalty>0pt
      <doublehyphendemerits>0pt<finalhyphendemerits>0pt
      <resize||0pt|-0.3bls|0pt|0.7bls>#<arg|1><resize||0pt|-0.3bls|0pt|0.7bls>>>>><assign|FV|<macro|<no-break>z@
      plus<fontdimen>3<font>minus<fontdimen>4<font>
      <discretionary|<Wrappedvisiblespacebox>|<Wrappedafterbreak>|<kern><fontdimen>2<font>>>>

      <Wrappedbreaksatspecials> <OriginalVerbatim*|<arg|1>,codes*=<Wrappedbreaksatpunct>>
    </macro>><assign|Verbatim|<macro|<Verbatim*|1>>>

    <assign|boxspacing|<macro|<kern><kvtcb@left@rule><kern><kvtcb@boxsep>>>

    <assign|prompt|<macro|1|2|3|4|<with|font-family|tt|<llap|<with|color|#2|[<arg|3>]:<space|3pt><arg|4>>><vspace|->
    >>>
  </hide-preamble>

  <doc-data|<doc-title|7_test>|<doc-date|<date|>>>

  <hypertarget|tests|<subsection|7. Tests><label|tests>>

  <hypertarget|pile-ou-face|<subsubsection|7.1 Pile ou
  face><label|pile-ou-face>>

  Pour des <math|X<rsub|1>,...,X<rsub|n>> identiquement distribuées à valeur
  dans <math|<around|{|0,1|}>>.

  Décrire une procédure de test de l'hypothèse
  <math|P*<around|(|X=1|)>=<frac|1|2>> contre son contraire.

  Voir aussi : <hlink|Cours_Tests_L2Rennes_VMonbet_2009.pdf|https://perso.univ-rennes1.fr/valerie.monbet/doc/cours/Cours_Tests_2009.pdf>

  <hypertarget|solution|<paragraph|Solution><label|solution>>

  <\itemize>
    <tightlist>

    <item>Hypothèse <math|\<cal-H\><rsub|0>> :
    <math|P*<around|(|X=1|)>=<frac|1|2>> avec le niveau <math|\<alpha\>>

    <item>Hypothères alternative <math|\<cal-H\><rsub|1>> :
    <math|P*<around|(|X=1|)>\<ne\><frac|1|2>>
  </itemize>

  Les valeurs de <math|X> étant dans <math|<around|{|0,1|}>>, le modèle
  statistique applicable est une loi de Bernouilli de paramètre <math|p>.
  L'hypothèse <math|\<cal-H\><rsub|0>> correspond à <math|p=<frac|1|2>>.

  <math|p> est aussi l'espérance <math|\<mu\>> de la loi de Bernouilli, et
  donc la moyenne empirique $ <wide|\<mu\>|^><em|n = <frac|1|n>
  <big|sum>>{i=1}^n X_i$ converge vers <math|p>.

  On a <math|n*<wide|\<mu\>|^><rsub|n>\<sim\>\<cal-B\><around|(|n,p|)>> avec
  <math|\<cal-B\>> <hlink|loi binomiale|https://fr.wikipedia.org/wiki/Loi_binomiale>
  dont la probabilité est :

  <\equation*>
    \<bbb-P\>*<around|(|X=k|)>=<choose|n|k>p<rsup|k>*<around|(|1-p|)><rsup|n-k>,k\<in\>0..*n
  </equation*>

  L'hypothèse <math|\<cal-H\><rsub|0>> est rejetée si <math|\<mu\>=0.5> n'est
  pas dans l'intervalle:

  <\equation*>
    <around|[|<frac|1|n>*<big|sum><rsub|i=0><rsup|n>X<rsub|i>+q<rsub|<frac|\<alpha\>|2>>,<frac|1|n>*<big|sum><rsub|i=0><rsup|n>X<rsub|i>+q<rsub|1-<frac|\<alpha\>|2>>|]>
  </equation*>

  q étant la fonction quantile de la loi binomiale.

  <hypertarget|variables-aluxe9atoires-gaussiennes-i.i.d-de-variance-sigma2|<subsubsection|7.2
  Variables aléatoires Gaussiennes i.i.d de variance
  <math|\<sigma\><rsup|2>>><label|variables-aluxe9atoires-gaussiennes-i.i.d-de-variance-sigma2>>

  Soient <math|X<rsub|1>,...,X<rsub|n>> des variables aléatoires i.i.d selon
  des lois gaussiennes de moyenne (inconnue) \<mu\> et de variance connue
  <math|\<sigma\><rsup|2>>, i.e. <math|X<rsub|i>\<sim\>\<cal-N\><around|(|\<mu\>,\<sigma\><rsup|2>|)>>.

  Décrire une procédure de test de l'hypothèse \<mu\> = 1 contre son
  contraire.

  <hypertarget|solution|<paragraph|Solution><label|solution>>

  <\itemize>
    <tightlist>

    <item>Hypothèse <math|\<cal-H\><rsub|0>> : <math|\<mu\>=1> avec le niveau
    <math|\<alpha\>>

    <item>Hypothères alternative <math|\<cal-H\><rsub|1>> :
    <math|\<mu\>\<nin\>I*C<rsub|\<alpha\>><around|(|1|)>>, avec
    <math|I*C<rsub|\<alpha\>><around|(|1|)>> l'intervalle de confiance de
    niveau <math|\<alpha\>>
  </itemize>

  L'estimateur de <math|\<mu\>> est : <math|<wide|\<mu\>|^><rsub|n>=<frac|1|n>*<big|sum><rsub|i=1><rsup|n>X<rsub|i>>
  qui a pour propriété :

  <\equation*>
    T<around|(|X<rsub|1>,...,X<rsub|n>|)>=<frac|<sqrt|n>|\<sigma\>>*<around|(|<wide|\<mu\>|^><rsub|n>-\<mu\>|)>\<sim\>\<cal-N\><around|(|0,1|)>
  </equation*>

  L'hypothèse <math|\<cal-H\><rsub|0>> est rejetée si <math|\<mu\>=1> n'est
  pas dans l'intervalle <math|<around|[|<wide|\<mu\>|^>+<frac|\<sigma\>|<sqrt|n>>*q<rsub|<frac|\<alpha\>|2>>,<wide|\<mu\>|^>+<frac|\<sigma\>|<sqrt|n>>*q<rsub|1-<frac|\<alpha\>|2>>|]>>
  utilisant les quantiles de la loi <math|\<cal-N\><around|(|0,1|)>>

  <hypertarget|variables-aluxe9atoires-gaussiennes-induxe9pendantes-de-variances-sigma<rsub|i>2|<subsubsection|7.3
  Variables aléatoires Gaussiennes indépendantes de variances
  <math|\<sigma\><rsub|i><rsup|2>>><label|variables-aluxe9atoires-gaussiennes-induxe9pendantes-de-variances-sigma>>

  Soient <math|X<rsub|1>,...,X<rsub|n>> des variables aléatoires
  indépendantes et distribuées selon des lois gaussiennes de moyenne
  (inconnue) \<mu\> et de variances connues <math|\<sigma\><rsub|i><rsup|2>>,
  i.e., <math|X<rsub|i>\<sim\>\<cal-N\><around|(|\<mu\>,\<sigma\><rsub|i><rsup|2>|)>>.

  Décrire une procédure de test de l'hypothèse \<mu\> = 1 contre son
  contraire.

  <hypertarget|solution|<paragraph|Solution><label|solution>>

  Le test est posé comme en 7.2, la statistique de test <math|T> est modifiée
  pour renormaliser les <math|X<rsub|i>>. En effet :

  <\equation*>
    \<forall\>i,<frac|1|\<sigma\><rsub|i>>*<around|(|X<rsub|i>-\<mu\>|)>\<sim\>\<cal-N\><around|(|0,1|)>
  </equation*>

  Soit la statistique de test :

  <\equation*>
    T<around|(|X<rsub|1>,...,X<rsub|n>|)>=<sqrt|n>*<big|sum><rsub|i=1><rsup|n><frac|1|\<sigma\><rsub|i>>*<around|(|X<rsub|i>-\<mu\>|)>\<sim\>\<cal-N\><around|(|0,1|)>
  </equation*>

  Qui permet d'établir un intervalle de rejet en utilisant les quantiles de
  <math|\<cal-N\><around|(|0,1|)>>

  <\tcolorbox|breakable, size=fbox, boxrule=1pt, pad at
  break*=1mm,colback=cellbackground, colframe=cellborder>
    \ <prompt|In|incolor||<boxspacing>> <Verbatim|commandchars=<next-line>{}|<new-line>>
  </tcolorbox>
</body>

<\initial>
  <\collection>
    <associate|font-base-size|11>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.1.1|?>>
    <associate|auto-4|<tuple|1.2|?>>
    <associate|auto-5|<tuple|1.2.1|?>>
    <associate|auto-6|<tuple|1.3|?>>
    <associate|auto-7|<tuple|1.3.1|?>>
    <associate|pile-ou-face|<tuple|1.1|?>>
    <associate|solution|<tuple|1.3.1|?>>
    <associate|tests|<tuple|1|?>>
    <associate|variables-aluxe9atoires-gaussiennes-i.i.d-de-variance-sigma2|<tuple|1.2|?>>
    <associate|variables-aluxe9atoires-gaussiennes-induxe9pendantes-de-variances-sigma|<tuple|1.3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>7. Tests
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|2tab>|1.1<space|2spc>7.1 Pile ou face
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|4tab>|Solution
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.15fn>>

      <with|par-left|<quote|2tab>|1.2<space|2spc>7.2 Variables aléatoires
      Gaussiennes i.i.d de variance <with|mode|<quote|math>|\<sigma\><rsup|2>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|4tab>|Solution
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.15fn>>

      <with|par-left|<quote|2tab>|1.3<space|2spc>7.3 Variables aléatoires
      Gaussiennes indépendantes de variances
      <with|mode|<quote|math>|\<sigma\><rsub|i><rsup|2>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|4tab>|Solution
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.15fn>>
    </associate>
  </collection>
</auxiliary>