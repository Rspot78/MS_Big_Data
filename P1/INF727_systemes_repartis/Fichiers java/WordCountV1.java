import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.io.*;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.charset.Charset;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Iterator;

/* Out of memory error:
 * Dans la run configuration : Ajouter l’option -Xms1000m aux VM arguments
 * pour avoir 1Go de mémoire pour la machine virtuelle java (qui exécute votre programme)*/

public class WordCountV1 {

	public HashMap<String, Integer> occurences;
	public String filename;
	public long startTime;

	public WordCountV1(String strFN) {
		occurences = new HashMap<String, Integer>();
		filename = strFN;
		startTime = System.currentTimeMillis();
	}

	public void readLines_new() {
		List<String> lines = null;
		long timespan[] = new long[2];
		try {
			lines = Files.readAllLines(Paths.get(filename), Charset.forName("UTF-8"));
		} catch (IOException e) {
			System.out.println("Erreur lors de la lecture de " + filename);
			System.exit(1);
		}
		System.out.println("Fin du chargement du fichier en memoire : "
				+ (System.currentTimeMillis() - startTime) / 1000 + " secondes");
		System.out.println("Liste de " + lines.size() + " lignes");
		for (String line : lines) {
			countWords(line);
		}
	}

	public void countWords(String strTxt) {
		Integer nb;
		for (String mot : strTxt.split(" ")) {

			if (mot.trim() != "" && mot.trim() != "\t") {
				nb = occurences.get(mot);
				if (nb == null)
					occurences.put(mot, 1);
				else
					occurences.put(mot, nb + 1);
			}
		}
	}

	private HashMap sortByValues() {
		occurences = this.sortByKeys();
		List list = new LinkedList(occurences.entrySet());
		Collections.sort(list, Collections.reverseOrder(new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getValue()).compareTo(((Map.Entry) (o2)).getValue());
			}
		}));

		HashMap sortedHashMap = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
			Map.Entry entry = (Map.Entry) it.next();
			sortedHashMap.put(entry.getKey(), entry.getValue());
		}
		return sortedHashMap;
	}

	private HashMap sortByKeys() {
		List list = new LinkedList(occurences.entrySet());
		// Defined Custom Comparator here
		Collections.sort(list, new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getKey()).compareTo(((Map.Entry) (o2)).getKey());
			}
		});

		HashMap sortedHashMap = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
			Map.Entry entry = (Map.Entry) it.next();
			sortedHashMap.put(entry.getKey(), entry.getValue());
		}
		return sortedHashMap;
	}

	public void print_dict(int nb, int typesort) {
		HashMap sortedHashMap = new LinkedHashMap();
		int count = 0;
		if (typesort == 1) {
			sortedHashMap = sortByKeys();
		} else {
			sortedHashMap = sortByValues();
		}
		for (Object key : sortedHashMap.keySet()) {
			System.out.println(key + " : " + sortedHashMap.get(key));
			count += 1;
			if (count == nb + 1) {
				break;
			}
		}
	}

	public static void main(String[] args) {
		long startTime;
		long timespan[] = new long[2];

		startTime = System.currentTimeMillis();
		WordCountV1 wc = new WordCountV1("/tmp/CC-MAIN-20170322212949-00140-ip-10-233-31-227.ec2.internal.warc.wet");
		wc.readLines_new();
		;

		timespan[0] = (System.currentTimeMillis() - startTime) / 1000;

		wc.print_dict(100, 2);
		timespan[1] = (System.currentTimeMillis() - startTime) / 1000;

		System.out.println("Durée du wordount : " + timespan[0] + " secondes");
		System.out.println("Durée du tri : " + (timespan[1] - timespan[0]) + " secondes");
		System.out.println("Durée totale : " + timespan[1] + " secondes");
	}

}
