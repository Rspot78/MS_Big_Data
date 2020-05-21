import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Master {

	
	// attributes
	
	// master command to be executed
	private String masterCommand;
	// checking of process activity
	private boolean checkProcess;
	// time to wait for a response from the programme
	private int waitingTime;
	// message to print if command worked properly
	private String successCommand;
	// message to print if command failed
	private String failureCommand;
	
	
	// constructors
	
	public Master(String masterCommand, boolean checkProcess, int waitingTime, String successCommand, String failureCommand) {
		this.masterCommand = masterCommand;
		this.checkProcess = checkProcess;
		this.waitingTime = waitingTime;
		this.successCommand = successCommand;
		this.failureCommand = failureCommand;
	}
	
	
	// methods
	
	// method that runs the master command and checks whether it executes correctly
	public boolean runAndCheck() {
		// extract words from the command
		String[] command = masterCommand.split(" ");
		// execute the command with processBuilder
		ProcessBuilder processBuilder = new ProcessBuilder(command);
		// redirect error stream in output stream
		processBuilder.redirectErrorStream(true);
		// declare the new process
		Process process = null;
		// try execution
		try {
			process = processBuilder.start();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		// if check process is activated, check whether the process is responding
		if (checkProcess) {
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			// initiate the programme as active and responding
			boolean active = true;
			boolean noResponse = false;
			while (active) {
				try {
					// waitfor returns True if the programme stops
					// so to get boolean True when programs continues, one needs 'not waitfor'
					boolean activeProcess = !process.waitFor(waitingTime, TimeUnit.SECONDS);
					// read standard output; if anything in there, go on
					if (reader.ready()) {
						// buffer is sending information
						while (reader.ready()) {
							// read outputs
							int c = reader.read();
							// then print
                            // System.out.print((char) c);
							}
					} else if(activeProcess) {
						// if nothing obtained during waiting time
						noResponse = true;
					}
					// check whether programme is still active, or terminated because it is done or does not respond
					active = activeProcess && !noResponse;
				} catch (IOException | InterruptedException e) {
					e.printStackTrace();
				}
			}
			// the process is terminated: check whether it was successful or not
			int ErrorProcess = process.exitValue();
			if (ErrorProcess == 0) {
				return true;
			} else {
				return false;
			}
		}
		return true;
	}
	
	
	// method that prints the programme output
	public void printOutput(boolean success) {
		if (success) {
			System.out.println(successCommand);
		} else {
			System.out.println(failureCommand);
		}
	}
	

	// method that runs the master programme of step 5.1
	public static void master1() {
		Master master = new Master("ls -al /tmp", true, 100, "", "");
		master.runAndCheck();
	}
	
	
	// method that runs the master programme of step 5.2
	public static void master2() {
		Master master = new Master("ls /jesuiunhero", true, 100, "", "");
		master.runAndCheck();
	}
	
	
	// method that runs the master programme of step 5.3
	public static void master3() {
		Master master = new Master("java -jar /home/romain/tmp/rlegrand/Slave.jar -2 0", true, 100, "", "");
		master.runAndCheck();
	}
	
	
	// method that runs the master programme of step 6.1
	public static void master4() {
		Master master = new Master("java -jar /home/romain/tmp/rlegrand/Slave.jar -1 0", true, 100, "", "");
		master.runAndCheck();
	}
	
	
	// method that runs the master programme of step 6.2
	public static void master5() {
		Master master = new Master("java -jar /home/romain/tmp/rlegrand/Slave.jar -1 0", true, 15, "Slave.jar successfully executed.", "Slave.jar failed to execute.");
		boolean success = master.runAndCheck();
		master.printOutput(success);
	}
	
	
	// method that runs the master programme of step 9.2
	public static void master6() throws IOException {
		// path to text file containing the machine names
	    String fileName = "/home/romain/tmp/pc_names.txt";
	    // read file and loop over machines
	    List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
	 	for (String line:content) {
	 	    // command to test connection
	 	    String command1 = "ssh rlegrand@" + line +" 'hostname'";
	 	    // declare new Master
	 	    Master master = new Master(command1, true, 3,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
	 	    // run process
	 	    boolean success1 = master.runAndCheck();
	 	    // print the outcome of the test
	 	    master.printOutput(success1);
	 	    // if the machine is responding, go on: run Slave.jar
	 	    if (success1) {
	 	    	// command to run Slave.jar
	 	    	String command2 = "ssh rlegrand@" + line + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar -2 0";
	 	    	// declare new Master
	 	    	Master master2 = new Master(command2, true, 10, "Slave.jar executed on " + line + ".", "Slave.jar failed to execute on " + line + ".");
	 	    	// run process
	 		    boolean success2 = master2.runAndCheck();
	 		    // print the outcome of the test
	 		    master2.printOutput(success2);
	 	    }
	 	}
	}
	
	
	
	// method that runs the master programme of step 10.1
	public static void master7() throws IOException {
		// path to text file containing the machine names
	    String fileName = "/home/romain/tmp/pc_names.txt";
	    // creation of list of responding machines
	    List<String> respondingMachines = new ArrayList<String>();
	    // read file and loop over machines
	    List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
	 	for (String line:content) {
	 	    // command to test connection
	 	    String command1 = "ssh rlegrand@" + line +" 'hostname'";
	 	    // declare new Master
	 	    Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
	 	    // run process
	 	    boolean success1 = master1.runAndCheck();
	 	    // print the outcome of the test
	 	    master1.printOutput(success1);
	 	    // if the machine is responding, write it on the list of responding machines
	 	   if (success1) {
	 		  respondingMachines.add(line);
	 	   }
	 	}
	 	// provide the number of splits
	 	int splitNumber = 3;
	 	// compute the number of responding machines
	 	int machineNumber = respondingMachines.size();
	 	// initiate the machine index
	 	int machineIndex = 0;
	    // loop over the splits
	 	for(int split = 0; split < splitNumber; split++) {
	 		// determine machine on which to send split
	 		String machine = respondingMachines.get(machineIndex);
	 		// System.out.println("the machine is " + machine);
	 		// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
    		String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
    		// declare new Master
    		Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
    		// run process
	    	boolean success2 = master2.runAndCheck();
	    	// print the outcome of the test
	 	    // master2.printOutput(success2);
	    	// then transfer split file; start with command to transfer split
	    	String command3 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
	    	// System.out.println("the command is " + command4);
	    	// declare new Master
	 	    Master master3 = new Master(command3, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
	 	    // run process
	 	    boolean success3 = master3.runAndCheck();
	 	    // print the outcome of the test
	 	    master3.printOutput(success3);
	 		// update index
	 		machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
	 	}   
	}

	
	
	// method that runs the master programme of step 10.2
	public static void master8() throws IOException {
		
		// PART 1: responding machines
		// path to text file containing the machine names
	    String fileName = "/home/romain/tmp/pc_names.txt";
	    // creation of list of responding machines
	    List<String> respondingMachines = new ArrayList<String>();
	    // read file and loop over machines
	    List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
	 	for (String line:content) {
	 	    // command to test connection
	 	    String command1 = "ssh rlegrand@" + line +" 'hostname'";
	 	    // declare new Master
	 	    Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
	 	    // run process
	 	    boolean success1 = master1.runAndCheck();
	 	    // print the outcome of the test
	 	    master1.printOutput(success1);
	 	    // if the machine is responding, write it on the list of responding machines
	 	   if (success1) {
	 		  respondingMachines.add(line);
	 	   }
	 	}
	 	
	 	// PART 2: splits
	 	// provide the number of splits
	 	int splitNumber = 3;
	 	// compute the number of responding machines
	 	int machineNumber = respondingMachines.size();
	 	// initiate the machine index
	 	int machineIndex = 0;
	 	
	 	// PART 3 creation of splits folders
	    // loop over the splits
	 	for(int split = 0; split < splitNumber; split++) {
	 		// determine machine on which to send split
	 		String machine = respondingMachines.get(machineIndex);
	 		// System.out.println("the machine is " + machine);
	 		// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
    		String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
    		// declare new Master
    		Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
    		// run process
	    	boolean success2 = master2.runAndCheck();
	    	// print the outcome of the test
	 	    // master2.printOutput(success2);
	    	
	    	// PART 4: transfer of split text files
	    	// then transfer split file; start with command to transfer split
	    	String command3 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
	    	// System.out.println("the command is " + command4);
	    	// declare new Master
	 	    Master master3 = new Master(command3, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
	 	    // run process
	 	    boolean success3 = master3.runAndCheck();
	 	    // print the outcome of the test
	 	    master3.printOutput(success3);
	 	    
	 	    // PART 5: creation of maps folders
	 	    // now create the map folder on the machine, only if it does not exist already
	 	    String command4 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
   		    // declare new Master
   		    Master master4 = new Master(command4, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
   		    // run process
	    	boolean success4 = master4.runAndCheck();
	 	    
	 	    // PART 6: update of machine index and end of loop
	 		// update index
	 		machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
	 	}   
	}
	
	
	
	// method that runs the master programme of step 10.3
	public static void master9() throws IOException, InterruptedException {
			
		// PART 1: responding machines
		// path to text file containing the machine names
		String fileName = "/home/romain/tmp/pc_names.txt";
		// creation of list of responding machines
		List<String> respondingMachines = new ArrayList<String>();
		// read file and loop over machines
		List<String> content=Files.readAllLines(Paths.get(fileName));
	    // loop over all machines in the list
		for (String line:content) {
			// command to test connection
		    String command1 = "ssh rlegrand@" + line +" 'hostname'";
		 	// declare new Master
		 	Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
		 	// run process
		 	boolean success1 = master1.runAndCheck();
		 	// print the outcome of the test
		 	master1.printOutput(success1);
		 	// if the machine is responding, write it on the list of responding machines
		    if (success1) {
		    	respondingMachines.add(line); 
		 	}
		 }
		 	
		 // PART 2: splits and machines
		 // provide the number of splits
		 int splitNumber = 3;
		 // compute the number of responding machines
		 int machineNumber = respondingMachines.size();
		 // initiate the machine index
		 int machineIndex = 0;
		 // determine the machine for each split
		 List<String> splitMachines = new ArrayList<String>();
		 // loop over splits
		 for(int split = 0; split < splitNumber; split++) {
		 	// add corresponding machine
		    splitMachines.add(respondingMachines.get(machineIndex));
		 	// update index
		 	machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
		 }
		 // initiate list of commands for incoming parallel Slave program
		 List<String> commandLines = new ArrayList<String>();
		 // loop over the splits
		 for(int split = 0; split < splitNumber; split++) {
		 	// determine machine on which to send split
		 	String machine = splitMachines.get(split);
		 		
			// PART 3 creation of splits folders
		 	// System.out.println("the machine is " + machine);
		 	// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
	    	String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
	    	// declare new Master
	    	Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
	    	// run process
		    boolean success2 = master2.runAndCheck();
		    // print the outcome of the test
		 	// master2.printOutput(success2);
		    	
		    // PART 4: creation of maps folders
		 	// now create the map folder on the machine, only if it does not exist already
		 	String command3 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
	   		// declare new Master
	   		Master master3 = new Master(command3, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
	   		// run process
		    boolean success3 = master3.runAndCheck();
		    	
		    // PART 5: transfer of split text files
		    // then transfer split file; start with command to transfer split
		    String command4 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
		    // System.out.println("the command is " + command4);
		    // declare new Master
		 	Master master4 = new Master(command4, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
		 	// run process
		 	boolean success4 = master4.runAndCheck();
		 	// print the outcome of the test
		 	master3.printOutput(success4);
		 	    
		 	// PART 6: list of commands for the incoming parallel map programme
		 	commandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 0 " + Integer.toString(split));
		 }
		 	
		 // PART 7: parallel maps
		 //System.out.println(commandLines);
		 // initiate list of processes
		 List<Process> processus = new ArrayList<Process>();
		 // loop over all the map commands
		 for(String command : commandLines) {
         // initiate new process
		 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
		 pb.redirectErrorStream(true);
		 // start the process and add to process list
		 processus.add(pb.start());
		 }
		 // now loop over processes and wait for each of them to complete
		 for(Process process : processus) {
			 process.waitFor();
		 }
		 // when done, display “MAP FINISHED!”
		 System.out.println("MAP FINISHED!");	
	}	
	
	
	
	
	// method that runs the master programme of step 11.1
		public static void master10() throws IOException, InterruptedException {
				
			// PART 1: responding machines
			// path to text file containing the machine names
			String fileName = "/home/romain/tmp/pc_names.txt";
			// creation of list of responding machines
			List<String> respondingMachines = new ArrayList<String>();
			// read file and loop over machines
			List<String> content=Files.readAllLines(Paths.get(fileName));
		    // loop over all machines in the list
			for (String line:content) {
				// command to test connection
			    String command1 = "ssh rlegrand@" + line +" 'hostname'";
			 	// declare new Master
			 	Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
			 	// run process
			 	boolean success1 = master1.runAndCheck();
			 	// print the outcome of the test
			 	master1.printOutput(success1);
			 	// if the machine is responding, write it on the list of responding machines
			    if (success1) {
			    	respondingMachines.add(line); 
			 	}
			 }
			 // on écrit un fichier avec les machines qui répondent
			 // chemin au fichier d'écriture 
			 String fileNameWrite = "/home/romain/tmp/rlegrand/machines.txt";
			 // ouverture du fichier d'écriture
			 FileWriter writer = new FileWriter(fileNameWrite, true);
			 // loop over responding machines
			 for(String machine : respondingMachines) {
				 // write machine name
				 writer.write(machine);
				 writer.write("\r\n");
			 }
			 writer.close();
			 // send the file to all responding machines
			 for(String machine : respondingMachines) {
				 String command0 = "scp /home/romain/tmp/rlegrand/machines.txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand";
			     // declare new Master
			     Master master0 = new Master(command0, true, 10, "machines.txt sent to " + machine , "failed to send machines.txt to " + machine );
			     // run process
				 boolean success0 = master0.runAndCheck();
				 // print the outcome of the test
				 master0.printOutput(success0);

			 }
			 writer.close();
	
			 // PART 2: splits and machines
			 // provide the number of splits
			 int splitNumber = 3;
			 // compute the number of responding machines
			 int machineNumber = respondingMachines.size();
			 // initiate the machine index
			 int machineIndex = 0;
			 // determine the machine for each split
			 List<String> splitMachines = new ArrayList<String>();
			 // loop over splits
			 for(int split = 0; split < splitNumber; split++) {
			 	// add corresponding machine
			    splitMachines.add(respondingMachines.get(machineIndex));
			 	// update index
			 	machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
			 }
			 // initiate list of commands for incoming parallel Slave program
			 List<String> commandLines = new ArrayList<String>();
			 // loop over the splits
			 for(int split = 0; split < splitNumber; split++) {
			 	// determine machine on which to send split
			 	String machine = splitMachines.get(split);
			 		
				// PART 3 creation of splits folders
			 	// System.out.println("the machine is " + machine);
			 	// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
		    	String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
		    	// declare new Master
		    	Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
		    	// run process
			    boolean success2 = master2.runAndCheck();
			    // print the outcome of the test
			 	// master2.printOutput(success2);
			    	
			    // PART 4: creation of maps folders
			 	// now create the map folder on the machine, only if it does not exist already
			 	String command3 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
		   		// declare new Master
		   		Master master3 = new Master(command3, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
		   		// run process
			    boolean success3 = master3.runAndCheck();
			    	
			    // PART 5: transfer of split text files
			    // then transfer split file; start with command to transfer split
			    String command4 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
			    // System.out.println("the command is " + command4);
			    // declare new Master
			 	Master master4 = new Master(command4, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
			 	// run process
			 	boolean success4 = master4.runAndCheck();
			 	// print the outcome of the test
			 	master3.printOutput(success4);
			 	
			 	// PART 6: list of commands for the incoming parallel map programme
			 	commandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 0 " + Integer.toString(split));
			 }
			 	
			 // PART 7: parallel maps
			 //System.out.println(commandLines);
			 // initiate list of processes
			 List<Process> processus = new ArrayList<Process>();
			 // loop over all the map commands
			 for(String command : commandLines) {
	         // initiate new process
			 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
			 pb.redirectErrorStream(true);
			 // start the process and add to process list
			 processus.add(pb.start());
			 }
			 // now loop over processes and wait for each of them to complete
			 for(Process process : processus) {
				 process.waitFor();
			 }
			 // when done, display “MAP FINISHED!”
			 System.out.println("MAP FINISHED!");	
		}	
	
		
		
		
		// method that runs the master programme of step 11.2
		public static void master11() throws IOException, InterruptedException {
				
			// PART 1: responding machines
			// path to text file containing the machine names
			String fileName = "/home/romain/tmp/pc_names.txt";
			// creation of list of responding machines
			List<String> respondingMachines = new ArrayList<String>();
			// read file and loop over machines
			List<String> content=Files.readAllLines(Paths.get(fileName));
		    // loop over all machines in the list
			for (String line:content) {
				// command to test connection
			    String command1 = "ssh rlegrand@" + line +" 'hostname'";
			 	// declare new Master
			 	Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
			 	// run process
			 	boolean success1 = master1.runAndCheck();
			 	// print the outcome of the test
			 	master1.printOutput(success1);
			 	// if the machine is responding, write it on the list of responding machines
			    if (success1) {
			    	respondingMachines.add(line); 
			 	}
			 }
			 // on écrit un fichier avec les machines qui répondent
			 // chemin au fichier d'écriture 
			 String fileNameWrite = "/home/romain/tmp/rlegrand/machines.txt";
			 // ouverture du fichier d'écriture
			 FileWriter writer = new FileWriter(fileNameWrite, true);
			 // loop over responding machines
			 for(String machine : respondingMachines) {
				 // write machine name
				 writer.write(machine);
				 writer.write("\r\n");
			 }
			 writer.close();
			 // send the file to all responding machines
			 for(String machine : respondingMachines) {
				 String command0 = "scp /home/romain/tmp/rlegrand/machines.txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand";
			     // declare new Master
			     Master master0 = new Master(command0, true, 10, "machines.txt sent to " + machine , "failed to send machines.txt to " + machine );
			     // run process
				 boolean success0 = master0.runAndCheck();
				 // print the outcome of the test
				 master0.printOutput(success0);

			 }
			 writer.close();
	
			 // PART 2: splits and machines
			 // provide the number of splits
			 int splitNumber = 3;
			 // compute the number of responding machines
			 int machineNumber = respondingMachines.size();
			 // initiate the machine index
			 int machineIndex = 0;
			 // determine the machine for each split
			 List<String> splitMachines = new ArrayList<String>();
			 // loop over splits
			 for(int split = 0; split < splitNumber; split++) {
			 	// add corresponding machine
			    splitMachines.add(respondingMachines.get(machineIndex));
			 	// update index
			 	machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
			 }
			 // initiate list of commands for incoming parallel Slave program
			 List<String> commandLines = new ArrayList<String>();
			 // loop over the splits
			 for(int split = 0; split < splitNumber; split++) {
			 	// determine machine on which to send split
			 	String machine = splitMachines.get(split);
			 		
				// PART 3 creation of splits folders
			 	// System.out.println("the machine is " + machine);
			 	// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
		    	String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
		    	// declare new Master
		    	Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
		    	// run process
			    boolean success2 = master2.runAndCheck();
			    // print the outcome of the test
			 	// master2.printOutput(success2);
			    	
			    // PART 4: creation of maps folders
			 	// now create the map folder on the machine, only if it does not exist already
			 	String command3 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
		   		// declare new Master
		   		Master master3 = new Master(command3, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
		   		// run process
			    boolean success3 = master3.runAndCheck();
			    	
			    // PART 5: transfer of split text files
			    // then transfer split file; start with command to transfer split
			    String command4 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
			    // System.out.println("the command is " + command4);
			    // declare new Master
			 	Master master4 = new Master(command4, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
			 	// run process
			 	boolean success4 = master4.runAndCheck();
			 	// print the outcome of the test
			 	master3.printOutput(success4);
			 	
			    // PART 6: creation of shuffle folders
			    // now create the shuffle folder on the machine, only if it does not exist already
			 	String command5 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/shuffles";
		   		// declare new Master
		   		Master master5 = new Master(command5, true, 10, "no shuffles folder on " + machine , "shuffles exists on " + machine );
		   		// run process
			    boolean success5 = master5.runAndCheck();
			    // print the outcome of the test
			 	master5.printOutput(success5);
			 	
			 	// PART 7: list of commands for the incoming parallel map programme
			 	commandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 0 " + Integer.toString(split));
			 }
			 	
			 // PART 8: parallel maps
			 //System.out.println(commandLines);
			 // initiate list of processes
			 List<Process> processus = new ArrayList<Process>();
			 // loop over all the map commands
			 for(String command : commandLines) {
	         // initiate new process
			 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
			 pb.redirectErrorStream(true);
			 // start the process and add to process list
			 processus.add(pb.start());
			 }
			 // now loop over processes and wait for each of them to complete
			 for(Process process : processus) {
				 process.waitFor();
			 }
			 // when done, display “MAP FINISHED!”
			 System.out.println("MAP FINISHED!");	
		}
		
		
		
		
		
		// method that runs the master programme of step 11.3
				public static void master12() throws IOException, InterruptedException {
						
					// PART 1: responding machines
					// path to text file containing the machine names
					String fileName = "/home/romain/tmp/pc_names.txt";
					// creation of list of responding machines
					List<String> respondingMachines = new ArrayList<String>();
					// read file and loop over machines
					List<String> content=Files.readAllLines(Paths.get(fileName));
				    // loop over all machines in the list
					for (String line:content) {
						// command to test connection
					    String command1 = "ssh rlegrand@" + line +" 'hostname'";
					 	// declare new Master
					 	Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
					 	// run process
					 	boolean success1 = master1.runAndCheck();
					 	// print the outcome of the test
					 	master1.printOutput(success1);
					 	// if the machine is responding, write it on the list of responding machines
					    if (success1) {
					    	respondingMachines.add(line); 
					 	}
					 }
					 // on écrit un fichier avec les machines qui répondent
					 // chemin au fichier d'écriture 
					 String fileNameWrite = "/home/romain/tmp/rlegrand/machines.txt";
					 // ouverture du fichier d'écriture
					 FileWriter writer = new FileWriter(fileNameWrite);
					 // loop over responding machines
					 for(String machine : respondingMachines) {
						 // write machine name
						 writer.write(machine);
						 writer.write("\r\n");
					 }
					 writer.close();
					 // send the file to all responding machines
					 for(String machine : respondingMachines) {
						 String command0 = "scp /home/romain/tmp/rlegrand/machines.txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand";
					     // declare new Master
					     Master master0 = new Master(command0, true, 10, "machines.txt sent to " + machine , "failed to send machines.txt to " + machine );
					     // run process
						 boolean success0 = master0.runAndCheck();
						 // print the outcome of the test
						 master0.printOutput(success0);

					 }
					 writer.close();
			
					 // PART 2: splits and machines
					 // provide the number of splits
					 int splitNumber = 3;
					 // compute the number of responding machines
					 int machineNumber = respondingMachines.size();
					 // initiate the machine index
					 int machineIndex = 0;
					 // determine the machine for each split
					 List<String> splitMachines = new ArrayList<String>();
					 // loop over splits
					 for(int split = 0; split < splitNumber; split++) {
					 	// add corresponding machine
					    splitMachines.add(respondingMachines.get(machineIndex));
					 	// update index
					 	machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
					 }
					 // initiate list of commands for incoming parallel map program
					 List<String> mapCommandLines = new ArrayList<String>();
					 // initiate list of commands for incoming parallel shuffle program
					 List<String> shuffleCommandLines = new ArrayList<String>();
					 // loop over the splits
					 for(int split = 0; split < splitNumber; split++) {
					 	// determine machine on which to send split
					 	String machine = splitMachines.get(split);
					 		
						// PART 3 creation of splits folders
					 	// System.out.println("the machine is " + machine);
					 	// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
				    	String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
				    	// declare new Master
				    	Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
				    	// run process
					    boolean success2 = master2.runAndCheck();
					    // print the outcome of the test
					 	// master2.printOutput(success2);
					    	
					    // PART 4: creation of maps folders
					 	// now create the map folder on the machine, only if it does not exist already
					 	String command3 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
				   		// declare new Master
				   		Master master3 = new Master(command3, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
				   		// run process
					    boolean success3 = master3.runAndCheck();
					    	
					    // PART 5: transfer of split text files
					    // then transfer split file; start with command to transfer split
					    String command4 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
					    // System.out.println("the command is " + command4);
					    // declare new Master
					 	Master master4 = new Master(command4, true, 10, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
					 	// run process
					 	boolean success4 = master4.runAndCheck();
					 	// print the outcome of the test
					 	master3.printOutput(success4);
					 	
					    // PART 6: creation of shuffle folders
					    // now create the shuffle folder on the machine, only if it does not exist already
					 	String command5 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/shuffles";
				   		// declare new Master
				   		Master master5 = new Master(command5, true, 10, "no shuffles folder on " + machine , "shuffles exists on " + machine );
				   		// run process
					    boolean success5 = master5.runAndCheck();
					    // print the outcome of the test
					 	master5.printOutput(success5);
					 	
					    // PART 7: creation of shufflesreceived folders
					    // now create the shuffle folder on the machine, only if it does not exist already
					 	String command6 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/shufflesreceived";
				   		// declare new Master
				   		Master master6 = new Master(command6, true, 10, "no shufflesreceived folder on " + machine , "shufflesreceived exists on " + machine );
				   		// run process
					    boolean success6 = master6.runAndCheck();
					    // print the outcome of the test
					 	master6.printOutput(success6);
					 	
					 	// PART 8: list of commands for the incoming parallel map and shuffle programmes
					 	mapCommandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 0 " + Integer.toString(split));
					 	shuffleCommandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 1 " + Integer.toString(split));
					 }
					 	
					 // PART 9: parallel maps
					 //System.out.println(commandLines);
					 // initiate list of processes
					 List<Process> mapProcessus = new ArrayList<Process>();
					 // loop over all the map commands
					 for(String command : mapCommandLines) {
			         // initiate new process
					 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
					 pb.redirectErrorStream(true);
					 // start the process and add to process list
					 mapProcessus.add(pb.start());
					 }
					 // now loop over processes and wait for each of them to complete
					 for(Process process : mapProcessus) {
						 process.waitFor();
					 }
					 // when done, display “MAP FINISHED!”
					 System.out.println("MAP FINISHED!");
					 
					 // PART 10: parallel shuffle
					 //System.out.println(commandLines);
					 // initiate list of processes
					 List<Process> shuffleProcessus = new ArrayList<Process>();
					 // loop over all the shuffle commands
					 for(String command : shuffleCommandLines) {
			         // initiate new process
					 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
					 pb.redirectErrorStream(true);
					 // start the process and add to process list
					 shuffleProcessus.add(pb.start());
					 }
					 // now loop over processes and wait for each of them to complete
					 for(Process process : shuffleProcessus) {
						 process.waitFor();
					 }
					 // when done, display “SHUFFLE FINISHED!”
					 System.out.println("SHUFFLE FINISHED!");
				}
	
	
	public static void master13() throws IOException, InterruptedException {
					
		// PART 1: responding machines
		// path to text file containing the machine names
		String fileName = "/home/romain/tmp/pc_names.txt";
		// creation of list of responding machines
		List<String> respondingMachines = new ArrayList<String>();
		// read file and loop over machines
		List<String> content=Files.readAllLines(Paths.get(fileName));
	    // loop over all machines in the list
		for (String line:content) {
			// command to test connection
			String command1 = "ssh rlegrand@" + line +" 'hostname'";
			// declare new Master
			Master master1 = new Master(command1, true, 5,  "machine " + line + " is responding.", "machine " + line + " does not respond.");
			// run process
			boolean success1 = master1.runAndCheck();
			// print the outcome of the test
			master1.printOutput(success1);
			// if the machine is responding, write it on the list of responding machines
			if (success1) {
				respondingMachines.add(line); 
			}
		}
		// on écrit un fichier avec les machines qui répondent
		// chemin au fichier d'écriture 
		String fileNameWrite = "/home/romain/tmp/rlegrand/machines.txt";
		// ouverture du fichier d'écriture
		FileWriter writer = new FileWriter(fileNameWrite);
		// loop over responding machines
		for(String machine : respondingMachines) {
			// write machine name
			writer.write(machine);
			writer.write("\r\n");
		}
		writer.close();
		// send the file to all responding machines
		for(String machine : respondingMachines) {
		    String command0 = "scp /home/romain/tmp/rlegrand/machines.txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand";
			// declare new Master
			Master master0 = new Master(command0, true, 10, "machines.txt sent to " + machine , "failed to send machines.txt to " + machine );
			// run process
			boolean success0 = master0.runAndCheck();
			// print the outcome of the test
			master0.printOutput(success0);
		}
			
		// PART 2: splits and machines
		// provide the number of splits
		int splitNumber = 3;
		// compute the number of responding machines
		int machineNumber = respondingMachines.size();
		// initiate the machine index
		int machineIndex = 0;
		// determine the machine for each split
		List<String> splitMachines = new ArrayList<String>();
		// loop over splits
		for(int split = 0; split < splitNumber; split++) {
			// add corresponding machine
			splitMachines.add(respondingMachines.get(machineIndex));
			// update index
			machineIndex = 0 + (machineIndex+1) * ((machineIndex<machineNumber-1)?1:0);
		}
		// initiate list of commands for incoming parallel map program
		List<String> mapCommandLines = new ArrayList<String>();
		// initiate list of commands for incoming parallel shuffle program
		List<String> shuffleCommandLines = new ArrayList<String>();
		// initiate list of commands for incoming parallel reduce program
		List<String> reduceCommandLines = new ArrayList<String>();
		// loop over the splits
		for(int split = 0; split < splitNumber; split++) {
			// determine machine on which to send split
			String machine = splitMachines.get(split);
					 		
			// PART 3 creation of splits folders
			// System.out.println("the machine is " + machine);
			// create splits folder (only if it does not exist already, to avoid overwriting existing splits)
			String command2 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/splits";
			// declare new Master
			Master master2 = new Master(command2, true, 10, "no splits folder on " + machine , "splits exists on " + machine );
			// run process
			boolean success2 = master2.runAndCheck();
			// print the outcome of the test
			// master2.printOutput(success2);
		    	
		    // PART 4: creation of maps folders
		 	// now create the map folder on the machine, only if it does not exist already
		 	String command3 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/maps";
	   		// declare new Master
	   		Master master3 = new Master(command3, true, 10, "no maps folder on " + machine , "maps exists on " + machine );
	   		// run process
		    boolean success3 = master3.runAndCheck();
		    	
		    // PART 5: transfer of split text files
		    // then transfer split file; start with command to transfer split
		    String command4 = "scp /home/romain/tmp/rlegrand/S" + Integer.toString(split) +  ".txt rlegrand@" + machine + ":/cal/homes/rlegrand/tmp/rlegrand/splits";
		    // System.out.println("the command is " + command4);
		    // declare new Master
		 	Master master4 = new Master(command4, true, 10000, "S" + Integer.toString(split) + ".txt transferred to " + machine + ".", "failed to transfer S" + Integer.toString(split) + ".txt to" + machine + ".");
		 	// run process
		 	boolean success4 = master4.runAndCheck();
		 	// print the outcome of the test
		 	master3.printOutput(success4);
		 	
		    // PART 6: creation of shuffle folders
		    // now create the shuffle folder on the machine, only if it does not exist already
		 	String command5 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/shuffles";
	   		// declare new Master
	   		Master master5 = new Master(command5, true, 10000, "no shuffles folder on " + machine , "shuffles exists on " + machine );
	   		// run process
		    boolean success5 = master5.runAndCheck();
		    // print the outcome of the test
		 	master5.printOutput(success5);
		 	
		    // PART 7: creation of shufflesreceived folders
		    // now create the shuffle folder on the machine, only if it does not exist already
		 	String command6 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/shufflesreceived";
	   		// declare new Master
	   		Master master6 = new Master(command6, true, 10000, "no shufflesreceived folder on " + machine , "shufflesreceived exists on " + machine );
	   		// run process
		    boolean success6 = master6.runAndCheck();
		    // print the outcome of the test
		 	master6.printOutput(success6);
		 	
		    // PART 8: creation of reduce folders
		    // now create the shuffle folder on the machine, only if it does not exist already
		 	String command7 = "ssh rlegrand@" + machine + " mkdir /cal/homes/rlegrand/tmp/rlegrand/reduce";
	   		// declare new Master
	   		Master master7 = new Master(command7, true, 10, "no reduce folder on " + machine , "reduce exists on " + machine );
	   		// run process
		    boolean success7 = master7.runAndCheck();
		    // print the outcome of the test
		 	master7.printOutput(success7);
		 		
		 	// PART 9: list of commands for the incoming parallel map and shuffle programmes
		 	mapCommandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 0 " + Integer.toString(split));
		 	shuffleCommandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 1 " + Integer.toString(split));
		 }
		 for (int index = 0; index < machineNumber; index++) {
			 String machine = respondingMachines.get(index);
			 reduceCommandLines.add("ssh rlegrand@" + machine + " java -jar /cal/homes/rlegrand/tmp/rlegrand/Slave.jar 2 0");
		 }
		
		 // PART 10: parallel maps
		 // initiate chrono
		 long mapChronoStart = System.currentTimeMillis();
		 // initiate list of processes
		 List<Process> mapProcessus = new ArrayList<Process>();
		 // loop over all the map commands
		 for(String command : mapCommandLines) {
         // initiate new process
		 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
		 pb.redirectErrorStream(true);
		 // start the process and add to process list
		 mapProcessus.add(pb.start());
		 }
		 // now loop over processes and wait for each of them to complete
		 for(Process process : mapProcessus) {
			 process.waitFor();
		 }
		 // terminate time count
		 long mapChronoEnd = System.currentTimeMillis();
		 // compute total sort time, in seconds
		 float mapChronoTotal = (mapChronoEnd - mapChronoStart) / (float) 1000;
		 // when done, display “MAP FINISHED!”
		 System.out.println("MAP FINISHED!");
		 
		 // PART 11: parallel shuffle
		 // initiate chrono
		 long shuffleChronoStart = System.currentTimeMillis();
		 // initiate list of processes
		 List<Process> shuffleProcessus = new ArrayList<Process>();
		 // loop over all the shuffle commands
		 for(String command : shuffleCommandLines) {
         // initiate new process
		 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
		 pb.redirectErrorStream(true);
		 // start the process and add to process list
		 shuffleProcessus.add(pb.start());
		 }
		 // now loop over processes and wait for each of them to complete
		 for(Process process : shuffleProcessus) {
			 process.waitFor();
		 }
		 // terminate time count
		 long shuffleChronoEnd = System.currentTimeMillis();
	     // compute total sort time, in seconds
		 float shuffleChronoTotal = (shuffleChronoEnd - shuffleChronoStart) / (float) 1000;
		 // when done, display “SHUFFLE FINISHED!”
		 System.out.println("SHUFFLE FINISHED!");
		 
		 // PART 12: parallel reduce
		 // initiate chrono
		 long reduceChronoStart = System.currentTimeMillis();
		 // initiate list of processes
		 List<Process> reduceProcessus = new ArrayList<Process>();
		 // loop over all the shuffle commands
		 for(String command : reduceCommandLines) {
	     // initiate new process
		 ProcessBuilder pb = new ProcessBuilder(command.split(" "));
		 pb.redirectErrorStream(true);
		 // start the process and add to process list
		 reduceProcessus.add(pb.start());
		 }
		 // now loop over processes and wait for each of them to complete
		 for(Process process : reduceProcessus) {
			 process.waitFor();
		 }
	   	 // terminate time count
		 long reduceChronoEnd = System.currentTimeMillis();
		 // compute total sort time, in seconds
		 float reduceChronoTotal = (reduceChronoEnd - reduceChronoStart) / (float) 1000;
		 // when done, display “REDUCE FINISHED!”
		 System.out.println("REDUCE FINISHED!");
		 
		 // final display of time count
		 System.out.println("");
		 System.out.println("Total time for the map phase: " + mapChronoTotal + " seconds.");
		 System.out.println("Total time for the shuffle phase: " + shuffleChronoTotal + " seconds.");
		 System.out.println("Total time for the reduce phase: " + reduceChronoTotal + " seconds.");
	 
	}			
				
					
	
			
	// main method
	public static void main(String[] args) throws InterruptedException, IOException {
			
		// step 5.1
		// master1();
		// step 5.2
		// master2();
		// step 5.3
		// master3();
		// step 6.1
		// master4();
		// step 6.2
		// master5();
		// step 9.2
		// master6();
		// step 10.1
		// master7();
		// step 10.2
		// master8();
		// step 10.3
		// master9();
		// step 11.1
		// master10();
		// step 11.2
		// master11();
		// step 11.3
		// master12();
		// step 12.3
		master13();
	}

}
