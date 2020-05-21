<div align="center"><h2>Installation et configuration de Cassandra sur EC2</h2></div>

<p align="justify">Notre cluster Cassandra est constitué de trois noeuds. Les trois noeuds correspondent à trois instances EC2 (AWS) appartenant à la même région (us-east-1) mais à trois zones de disponibilité différentes (us-east-1a, us-east-1b, us-east-1c).</p>

<p align='justify'>Les trois instances EC2 ont été créées directement sur la plateforme AWS. Nous avons choisi des images AMI Ubuntu 18.04.</p>

<p align='justify'>Afin d'installer et de configurer Cassandra sur les trois instances, nous nous sommes connectés à chacune d'entre elles.</p>

<h3>Connexion SSH aux instances EC2</h3>

Dans le terminal, la commande est la suivante : 

``` bash
ssh -v ubuntu@ec2-34-224-37-57.compute-1.amazonaws.com -i ~/Downloads/gdeltKeyPair.pem
```

<h3>Configurations préliminaires</h3>

Une fois connecté, les étapes suivantes ont été réalisées :
<ul>
  <li>Ajout du repo webupd8 au système : </li>
  </ul>

```bash 
sudo add-apt-repository ppa:webupd8team/java
```

<ul>
  <li>Mise à jour des packages : </li>
  </ul>

```bash 
sudo apt-get update
```

<ul>
<li>Installation de java : </li>
  </ul>

```bash 
sudo apt install openjdk-8-jre-headless 
```

<p align='justify'>Ensuite, il a fallu ajouter des variables d'environnement afin d'indiquer au système le chemin du répertoire d'exécution. Pour cela, nous avons modifié le fichier .bashrc de la façon suivante :</p>

<ul>
  <li>Ouverture du fichier .bashrc avec un éditeur de texte (ici Vim) : </li>
  </ul>

```bash 
vi ~/.bashrc
```

<ul>
  <li>Ajout des variables d'environnement : </li>
  </ul>

```bash 
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export JRE_HOME=$JAVA_HOME/jre
export PATH=$PATH:$JAVA_HOME/bin:$JAVA_HOME/jre/bin
```

Ne pas oublier de les charger avec la commande :

```bash 
source ~/.bashrc
```

Ces commandes ont été effectuées sur chaque instance. 

<h3>Installation de Cassandra</h3>

<ul>
  <li>Téléchargement du fichier d'installation (janvier 2020) : </li>
  </ul>

```bash 
wget http://archive.apache.org/dist/cassandra/3.11.3/apache-cassandra-3.11.3-bin.tar.gz
```

<ul> 
  <li>Décompression du fichier zip : </li>
  </ul>

```bash 
tar -xvf apache-cassandra-3.11.3-bin.tar.gz
```

<ul>
  <li>Suppression du fichier zip : </li>
  </ul>

```bash 
rm apache-cassandra-3.11.3-bin.tar.gz
```

<ul>
  <li>Suppression du fichier zip : </li>
  </ul>

```bash 
rm apache-cassandra-3.11.3-bin.tar.gz
```

<p align='justify'>Cassandra est désormais installé dans le /home de l'instance. Ces commandes ont été effectuées sur chaque instance avant de passer à l'étape suivante.</p>

<h3>Configuration de Cassandra</h3>

<ul>
  <li>Positionnement dans le dossier de configuration de Cassandra : </li>
  </ul>

```bash 
cd apache-cassandra-3.11.3/conf/
```

Nous avons configuré deux fichiers : cassandra.yaml et cassandra-rackdc.properties. 

<b>Configuration du fichier cassandra.yaml</b>

<ul>
  <li>Ouverture du fichier cassandra.yaml : </li>
  </ul>

```bash 
vi cassandra.yaml
```

<ul>
  <li>Modification des éléments suivants : </li>
  </ul>

```bash 
seeds : "ip_noeud1,ip_noeud2,ip_noeud3"
endpoint_snitch : Ec2Snitch
listen_address : ip privée du noeud en question sans guillemets
rpc_address : ip privée du noeud en question sans guillemets
```

<p align='justify'>Ec2Snitch est utilisé car nous développons le cluster dans une seule région. Comme notre facteur de réplication 
était de 3 dans la région USA Est (Virginie du Nord), us-east-1, alors le paramètre Ec2Snitch permet la réplication des données 
sur les zones de disponibilité dans us-east-1. Chaque opération d'écriture a été répliquée sur les nœuds de nos trois zones de disponibilité (us-east-1a, us-east-1b et us-east-1c). 
Chaque zone de disponibilité était un rack différent.</p>

<b>Configuration du fichier cassandra-rackdc.properties</b>

Les éléments relatifs aux datacenters (dc=dc1) et aux racks (rack=rack1) ont été commentés.

<p align='justify'>Ces étapes ont été réalisées sur chaque noeud avant de démarrer Cassandra.</p>

<h3>Démarrage de Cassandra et vérification de la communication entre les noeuds</h3>

Sur chaque noeud, nous avons effectué les opérations suivantes :

<ul>
  <li>Positionnement dans le dossier /bin de Cassandra : </li>
  </ul>

```bash 
cd apache-cassandra-3.11.3/bin/
```

<ul>
  <li>Exécution de Cassandra : </li>
  </ul>

```bash 
./cassandra
```

<p align='justify'>Cassandra a démarré et la présence du mot "HANDSHAKE" dans certaines lignes nous a indiqué que les noeuds communiquaient. </p>

Nous avons également vérifié l'état des différents noeuds avec la commande :

```bash 
./nodetool status
```

