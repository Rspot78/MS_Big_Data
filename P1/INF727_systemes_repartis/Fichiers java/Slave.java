import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Slave {
	
	
	// attributes
	
	// type of programme being executed
	private String slaveType;
	// file number on which the programme is executed
	private String fileNumber;
		
	
	// constructors
	
	public Slave(String slaveType, String fileNumber) {
		this.slaveType = slaveType;
		this.fileNumber = fileNumber;
	}
	
	
	// methods
	
	// method that runs the slave programme
	public static void slaveProgramme(Slave sl) throws InterruptedException, IOException {
		sl.runProgramme();
	}
	
	
	// method that runs the adequate slave programme
	public void runProgramme() throws InterruptedException, IOException {
		// if the programme type is "-2", run the simple calculation programme
		if (slaveType.equals("-2")) {
			calculation();
		// if the programme type is "-1", run the programme of calculation with waiting time
		} else if (slaveType.equals("-1")) {
			calculationWait();
		// if the programme type is "0", run the map programme
		} else if (slaveType.equals("0")) {
			mapper();
		// if the programme type is "1", run the shuffle programme
		} else if (slaveType.equals("1")) {
			shuffle();
		// if the programme type is "1", run the reduce programme
		} else if (slaveType.equals("2")) {
		reduce();
	    }
	}
	
	
	// method that executes the simple calculation, without waiting
	public static void calculation() {
		int sum=3+5;
		System.out.println("The addition of 3 and 5 yields " + sum + ".");
	}
	
	
	// method that executes the simple calculation, with waiting time
	public static void calculationWait() throws InterruptedException {
		int sum=3+5;
		Thread.sleep(10000);
		System.out.println("The addition of 3 and 5 yields " + sum + ".");
	}	
	
	
	// method that executes the map sequence of the programme
	public void mapper() throws IOException {
		// path to read file Sx.txt
		String fileNameRead = "/cal/homes/rlegrand/tmp/rlegrand/splits/S" + fileNumber + ".txt";
		// path to write file UMx.txt
		String fileNameWrite = "/cal/homes/rlegrand/tmp/rlegrand/maps/UM" + fileNumber + ".txt";
		// opening write file
		FileWriter writer = new FileWriter(fileNameWrite, true);
		// reading input file
		List<String> content=Files.readAllLines(Paths.get(fileNameRead));
		// loop on file lines
		for (String line:content) {
			// split line elements
			String[] words=line.split(" ");
			// write in text file
			for (String word:words) {
				writer.write(word + " 1");
				writer.write("\r\n");
			}
		}
		writer.close();
	}

		
		
	// method that executes the shuffle sequence of the programme
	public void shuffle() throws IOException, InterruptedException {
		// obtain first the name of this machine
		String thisMachine = InetAddress.getLocalHost().getHostName();
		// obtain then the list of all working machines on the network
		String machineFile = "/cal/homes/rlegrand/tmp/rlegrand/machines.txt";
		List<String> machines = Files.readAllLines(Paths.get(machineFile));
		// get the number of machines on the network
		int numberMachines = machines.size();
		// initiate an arraylist of keys
		List<String> keys = new ArrayList<String>();
		// initiate an arraylist of commands
		List<String> commands = new ArrayList<String>();
		// path to read file UMx.txt
		String fileNameRead = "/cal/homes/rlegrand/tmp/rlegrand/maps/UM" + fileNumber + ".txt";
		// read input file
		List<String> content=Files.readAllLines(Paths.get(fileNameRead));
		// loop over elements
		for (String line:content) {
			// extract key
			String key = line.split(" ",2)[0];
			// get hash for the key
			long hash = Integer.toUnsignedLong(key.hashCode());
			// name of write file
			String fileNameWrite = "/cal/homes/rlegrand/tmp/rlegrand/shuffles/" + Long.toString(hash) + "-" + thisMachine + ".txt";
			// open write file
			FileWriter writer = new FileWriter(fileNameWrite, true);
			// write line
			writer.write(line);
			writer.write("\r\n");
			// close writer
			writer.close();
			// check if the key appears for the first time
			// if yes, prepare a command to send the text file to another machine
			if (!keys.contains(key)) {
				// add key to the list
				keys.add(key);
				// obtain the index of machine to which the file is sent
				long machineIndex = hash % numberMachines;
				// get the corresponding name
				String otherMachine = machines.get((int) machineIndex);
				// if the source and target machines are not the same, use scp
				if (!thisMachine.equals(otherMachine)) {
					// part 1 of scp command
					String part1 = "scp /cal/homes/rlegrand/tmp/rlegrand/shuffles/" + Long.toString(hash) + "-" + thisMachine + ".txt";
					// part 2 of scp command
					String part2 = "rlegrand@" + otherMachine + ":/cal/homes/rlegrand/tmp/rlegrand/shufflesreceived";
					// create the scp command to send the file from this machine to other machine
					String command = part1 + " " + part2;
					commands.add(command);
				// if the file remains on the same machine, just use cp	
				} else {
					// part 1 of cp command
					String part1 = "cp /cal/homes/rlegrand/tmp/rlegrand/shuffles/" + Long.toString(hash) + "-" + thisMachine + ".txt";
					// part 2 of cp command
					String part2 = "/cal/homes/rlegrand/tmp/rlegrand/shufflesreceived";
					// create the cp command to send the file from this machine to other machine
					String command = part1 + " " + part2;
					commands.add(command);
				}
			}
		}
		for(String command:commands) {
		    // now execute the commands to transfer the text files
			ProcessBuilder pb = new ProcessBuilder(command.split(" "));
			Process p = pb.start();
			p.waitFor();
		}
	}
		
	
	
	// method that executes the reduce sequence of the programme
	public void reduce() throws IOException, InterruptedException {
		
		// create two hashmaps for incoming files: one with hash/word, the other with hash/count
		HashMap<String,String> words = new HashMap<String,String>();
		HashMap<String, Integer> count = new HashMap<String, Integer>();
		// obtain list of all files in shufflesreceived
		File folder = new File("/cal/homes/rlegrand/tmp/rlegrand/shufflesreceived");
        File[] files = folder.listFiles();
        // loop over files
        for (File file : files){
        	// extract (hash) key in file name
        	String hashKey = file.getName().split("-",2)[0];
        	// full path to file
    		String fileNameRead = "/cal/homes/rlegrand/tmp/rlegrand/shufflesreceived/" + file.getName();
    		// read file
    		List<String> content=Files.readAllLines(Paths.get(fileNameRead));
    		// get file length;
    		int fileCount = content.size(); 
    		// get file word
    		String fileWord = content.get(0).split(" ",2)[0];
    		// if hashkey is not yet in hashmaps, insert it
        	if (!words.containsKey(hashKey)) {
        		words.put(hashKey, fileWord);
        		count.put(hashKey, fileCount);
			// otherwise, just increment count by fileCount
			} else {
				count.put(hashKey, count.get(hashKey) + fileCount);
			}
        }
        // Now loop over hashkeys, create txt files and enter the pair word, count
        for (String key : words.keySet()) {
        	// get hashkey
        	String hkey = key;
        	// get word
        	String hword = words.get(hkey);
        	// get count
        	String hcount = String.valueOf(count.get(hkey));
        	// name of write file
        	String fileNameWrite = "/cal/homes/rlegrand/tmp/rlegrand/reduce/" + hkey + ".txt";
        	// open write file
        	FileWriter writer = new FileWriter(fileNameWrite);
        	// write line
        	writer.write(hword + " " + hcount);
        	// close writer
        	writer.close();
        }
	}
	
	
	
	
		
	// main method
	public static void main(String[] args) throws InterruptedException, IOException {
		
		// default value, if programme is run with no arguments
		String arg0 = "-2";
		String arg1 = "0";
		// if arguments were actually passed, take them
		if(args.length != 0) {
			arg0 = args[0];
			arg1 = args[1];
		}
		
		// declare new Slave
		Slave slave = new Slave(arg0, arg1);
		
		// run the slaveProgramme method
		slaveProgramme(slave);
	}
}
