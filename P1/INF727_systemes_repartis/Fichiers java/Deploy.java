import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Deploy {
	
	
	// attributes
	
	// master command to be executed
	private String deployCommand;
	// checking of process activity
	private boolean checkProcess;
	// time to wait for a response from the programme
	private int waitingTime;
	// message to print if command worked properly
	private String successCommand;
	// message to print if command failed
	private String failureCommand;
	
	
	// constructors
	
	public Deploy(String deployCommand, boolean checkProcess, int waitingTime, String successCommand, String failureCommand) {
		this.deployCommand = deployCommand;
		this.checkProcess = checkProcess;
		this.waitingTime = waitingTime;
		this.successCommand = successCommand;
		this.failureCommand = failureCommand;
	}
	
	
	// methods
	
	// method that runs the deploy command and checks whether it executes correctly
	public boolean runAndCheck() {
		// extract words from the command
		String[] command = deployCommand.split(" ");
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
		// if check process is not activated, assume the process is responding
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
							}
					} else if(activeProcess) {
						// if nothing obtained during waiting time
						noResponse = true;
						process.destroy();
					}
					// check whether programme is still active, or terminated because it is done or does not respond
					active = activeProcess && !noResponse;
				} catch (IOException | InterruptedException e) {
					e.printStackTrace();
				}
			}
			return !noResponse;
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
	

	// method that runs the deploy programme of step 7.1
	public static void deploy1() throws IOException {
		// path to text file containing the machine names
    	String fileName = "/home/romain/tmp/pc_names.txt";
    	// read file and loop over machines
    	List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
		for (String line:content) {
    		// command to test connection
    		String command = "ssh rlegrand@" + line +" 'hostname'";
    		// declare new Deploy
    		Deploy deploy = new Deploy(command, true, 3, "machine " + line + " is responding.", "machine " + line + " does not respond.");
    		// run process
    		boolean success = deploy.runAndCheck();
    		// print the outcome of the test
    		deploy.printOutput(success);
		}
	}
	
	
	// method that runs the deploy programme of step 7.2
	public static void deploy2() throws IOException {
		// path to text file containing the machine names
	    String fileName = "/home/romain/tmp/pc_names.txt";
	    // read file and loop over machines
	    List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
		for (String line:content) {
	    	// command to test connection
	    	String command1 = "ssh rlegrand@" + line +" 'hostname'";
	    	// declare new Deploy
	    	Deploy deploy1 = new Deploy(command1, true, 3, "machine " + line + " is responding.", "machine " + line + " does not respond.");
	    	// run process
	    	boolean success1 = deploy1.runAndCheck();
	    	// print the outcome of the test
	    	deploy1.printOutput(success1);
	    	// if the machine is responding, go on: create directory and send Slave.jar
	    	if (success1) {
	    		// command to create directory
	    		String command2 = "ssh rlegrand@" + line + " mkdir -p /cal/homes/rlegrand/tmp/rlegrand";
	    		// declare new Deploy
	    		Deploy deploy2 = new Deploy(command2, true, 10, "directory created on " + line + ".", "failed to create directory on " + line + ".");
	    		// run process
		    	boolean success2 = deploy2.runAndCheck();
		    	// print the outcome of the test
		    	deploy2.printOutput(success2);
		    	// if the directory is created, go on: create directory and send Slave.jar
		    	if (success2) {
		    		// command to transfer slave.jar
			    	String command3 = "scp /home/romain/tmp/rlegrand/Slave.jar rlegrand@" + line + ":/cal/homes/rlegrand/tmp/rlegrand";
			    	// declare new Deploy
		    		Deploy deploy3 = new Deploy(command3, true, 10, "Slave.jar transfered to " + line + ".", "failed to transfer Slave.jar to " + line + ".");
		    		// run process
			    	boolean success3 = deploy3.runAndCheck();
			    	// print the outcome of the test
			    	deploy3.printOutput(success3);
		    	}
	    	}
		}
	}
	
	
	// main method
	public static void main(String[] args) throws InterruptedException, IOException {
			
		// step 7.1
		// deploy1();
		// step 7.2
		deploy2();
	}	

}
