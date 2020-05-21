import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class WordCount {
	
	
	// attributes
	
	// unsorted index of document words
	private HashMap<String,Integer> unsortedIndex;
	// sorted index of document words
	private LinkedHashMap<String,Integer> sortedIndex;
	// path to filePath with words to be extracted
	private String filePath;
	// sort type (0: no sort; 1: sequential sort; 2: sequential, then alphabetical sort)
	private int sortType;
	// result display type (-1: all entries, 0 : no display, n>0: n first entries)
	private int printType;
	// time counter for running application
	private long timeCounter;
	
	
	// constructors
	
	public WordCount(String filePath, int sortType, int printType) {
		unsortedIndex = new HashMap<String,Integer>();
		sortedIndex = new LinkedHashMap<String,Integer>();
		this.filePath = filePath;
		this.sortType = sortType;
		this.printType = printType;
		timeCounter = 0;
	}
	
	
	// methods
	
	// method to compute sequential count, no sort
	public void createIndex() throws IOException{
		// start time count
		long chronoStart = System.currentTimeMillis();
		// read data and transfer to list
		List<String> content=Files.readAllLines(Paths.get(filePath));
		// loop over file lines
		for (String line:content) {
			// split line elements and insert into index
			splitLine(line);	
		}
		// terminate time count
		long chronoEnd = System.currentTimeMillis();
		// compute total count time, in seconds
		float chronoTotal = (chronoEnd - chronoStart) / (float) 1000 ;
		// update time counter
		timeCounter += chronoEnd - chronoStart;
		// result print
		System.out.println("Word index successfully generated. The time count for word occurences is " + chronoTotal + " seconds.");
	}
	
	
	// method to split lines from input file and insert obtained words into index
	public void splitLine(String line) {
		// split the line to obtain words
		String[] words=line.split(" ");	
		// insert words into index
		insertIndex(words);
	}
	
	
	// method to insert words into the index
	public void insertIndex(String[] words) {
		for (String word:words) {
			// if word does not exist in index, insert it
			if (!unsortedIndex.containsKey(word)) {
				unsortedIndex.put(word, 1);
				// otherwise, just increment count by one
			} else {
				unsortedIndex.put(word, unsortedIndex.get(word)+1);
			}	
		}
	}
	
	
	// method to sort index entries
	public void sortIndex() {
		// start time counter
		long chronoStart = System.currentTimeMillis();
		// if sequential sort only, sort by decreasing occurrences
		if (sortType == 1) {
			// create temporary data list
			List<Entry<String, Integer>> list = new ArrayList<Entry<String, Integer>>(unsortedIndex.entrySet());
			// then sort
		    Collections.sort(list, (a, b) -> {
		    	int comparator = -a.getValue().compareTo(b.getValue());
		    	return comparator;
		    });
		    // finally, insert into into sortedIndex
		    for (HashMap.Entry<String, Integer> entry: list) {
		    	sortedIndex.put(entry.getKey(), entry.getValue());
		    }
		// if sequential and alphabetical sort, sort decreasing for occurrences, then increasing for alphabetical
		} else if (sortType == 2) {
			// create temporary data list
			List<Entry<String, Integer>> list = new ArrayList<Entry<String, Integer>>(unsortedIndex.entrySet());
			// then sort for occurrences
		    Collections.sort(list, (a, b) -> {
		    	int comparator = -a.getValue().compareTo(b.getValue());
		        if (comparator != 0) {
		        	return comparator;
		        // if equality of occurrences, sort alphabetical
		        } else {
		        	return a.getKey().compareTo(b.getKey());
		        }
		    });
		 // finally, insert into into sortedIndex
		    for (HashMap.Entry<String, Integer> entry : list) {
		    	sortedIndex.put(entry.getKey(), entry.getValue());
		    }
		}
		// otherwise, no sort: display unsortIndex as it is
		else {
			for (Map.Entry <String, Integer> entry : unsortedIndex.entrySet()) {
		    	sortedIndex.put(entry.getKey(), entry.getValue());
			}
		}
		// terminate time count
		long chronoEnd = System.currentTimeMillis();
		// compute total sort time, in seconds
		float chronoTotal = (chronoEnd - chronoStart) / (float) 1000;
		// update time counter
		timeCounter += chronoEnd - chronoStart;
		// result print
		System.out.println("Sorting successfully operated. The time count for sort operations is " + chronoTotal + " seconds.");
		
	}
	
	
	// method to display index results and final program time
	public void printIndex() {
		// calculate total time
		float finalTime = timeCounter / (float) 1000;
		// print
		System.out.println("Programme ran successfully. Total running time is " + finalTime + " seconds.");
		int count = 0;
		int maxCount = 0;
		// if no display, skip
		if (printType == 0) {
			return;
		// if display of a given number of entries only, restrict to these first n entries
		} else if (printType > 0) {
			maxCount = Math.min(printType, sortedIndex.size());
		// if display of all entries, consider length of sortedIndex
		} else {
			maxCount = sortedIndex.size();
		}
		// loop over keys and associated values in sortedIndex
        for (String name : sortedIndex.keySet()) {
        	// print key and value
            System.out.println(name + "  " + sortedIndex.get(name));
            count++;
            // stop when maxCount is reached
            if (count >= maxCount) {
            	System.out.println("  ");
            	break;
            }
        }	
	}
	
	// method which groups the set of methods for index creation, sort, and display
	public static void wordCount(WordCount wc) throws IOException {
		wc.createIndex();
		wc.sortIndex();
		wc.printIndex();
	}
	
	
	// main method
	public static void main(String[] args) throws IOException {
		
		// step 1.1: word count without sort
		WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/input.txt", 0, -1);
		// step 1.2: word count with sort on occurrences only
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/input.txt", 1, -1);
		// step 1.3: word count with sort on occurrences, then alphabetical
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/input.txt", 2, -1);
		// step 1.4: wordcount test on "code forestier de Mayotte"
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/forestier_mayotte.txt", 2, -1);
		// step 1.5: first 50 words of the "code de la déontologie de la police nationale"
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/deontologie_police_nationale.txt", 2, 50);
		// step 1.6: first 50 words of the "code du domaine public fluvial "
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/domaine_public_fluvial.txt", 2, 50);		
		// step 1.7: first 50 words of the "code de la santé publique"
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/sante_publique.txt", 2, 50);
		// step 1.8: first 50 words of the "code de la santé publique"
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/sante_publique.txt", 2, 50);
		// step 1.9: work on bigger files
		// WordCount wordCount = new WordCount("/home/romain/Systemes_Repartis/CC-MAIN-20170322212949-00140-ip-10-233-31-227.ec2.internal.warc.wet", 2, 50);
		
		// run the wordCount method
		wordCount(wordCount);
	}
}
