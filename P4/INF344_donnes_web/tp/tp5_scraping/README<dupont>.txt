
1) rôle général d'une méthode comme linksfilter

Les pages HTML explorées par les méthodes de crawling contiennent souvent un très grand nombre de liens. Parmi ces liens, certains sont inutiles en ce qui concerne la tâche
que le crawler veut effectuer. Par exemple, des liens de contact email vers le propriétaire du site ou de création de compte (cf "contact" et "register" dans la liste des liens
à éliminer dans la méthode linksfilter) sont peu susceptibles de transporter vers des pages contenant du contenu intéressant, car ce sont des pages essentiellement administratives.
On peut de même vouloir éviter de répéter les urls dans la liste finale obtenue (inutile d'explorer la même page plusieurs fois, à part pour gâcher du temps de calcul!), ou de
n'être redirigé que vers un autre moteur de recherche (pages "research" et "search" à éliminer dans la fonction linkfilter).

De façon générale, on peut dire qu'une fonction comme linksfilter sert à s'assurer que les pages crawlées renvoyées contiendront de l'information utile, pertinante, et non redondante.


2) rôle général d'une méthode comme contentfilter

Une méthode comme contentfilter permet à une application de crawling très spécialisée de détecter automatiquement si la page explorée contient du contenu utile pour la tâche à remplir.
De façon équivalente, on peut dire que ce type de méthode élimine les pages qui ne fournissent pas les informations adéquates pour la recherche menée. Dans le cas de contentfilter,
on veut par exemple s'assurer que la page explorée contiendra à la fois le nom de l'inventeur, le titre de l'invention, et le numéro de soumission du brevet. Une page ne contenant 
pas l'intégralité de ces informations sera considérée comme non satisfaisante et écartée.

De façon générale, on peut dire qu'une fonction comme contentfilter sert à s'assurer que les pages crawlées renvoyées contiendront toute l'information souhaitée et non un contenu lacunaire.

3) limites du code proposé à l'étape 6

Le code proposé est très spécialisé et requiert de maîtriser parfaitement à la fois la construction du site à crawler et la structure de l'information à rechercher.
Le code proposé pourrait de plus manquer sa cible lors de mises à jour. Par exemple, le code cible l'expression "Inventors:". Que se passera-t-il si demain la typologie du site change légèrement?
Par exemple, si "Inventors:" est remplacé par "inventors:" ou "inventors " (sans le semicolon)? Le crawler deviendra incapable de trouver les noms en question.
Enfin, ce code est très spécialisé: il faudra potentiellement l'adapter chaque fois qu'on ira crawler un site différent, ou voudra mener une recherche différente.














