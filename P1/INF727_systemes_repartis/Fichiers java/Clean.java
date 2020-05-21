import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Clean {
	
	
	// attributes
	
	// master command to be executed
	private String cleanCommand;
	// checking of process activity
	private boolean checkProcess;
	// time to wait for a response from the programme
	private int waitingTime;
	// message to print if command worked properly
	private String successCommand;
	// message to print if command failed
	private String failureCommand;
	
	
	// constructors
	
	public Clean(String cleanCommand, boolean checkProcess, int waitingTime, String successCommand, String failureCommand) {
		this.cleanCommand = cleanCommand;
		this.checkProcess = checkProcess;
		this.waitingTime = waitingTime;
		this.successCommand = successCommand;
		this.failureCommand = failureCommand;
	}
	
	
	// methods
	
	// method that runs the clean command and checks whether it executes correctly
	public boolean runAndCheck() {
		// extract words from the command
		String[] command = cleanCommand.split(" ");
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
	
	
	// method that runs the clean programme of step 8.1
	public static void clean() throws IOException {
		// path to text file containing the machine names
	    String fileName = "/home/romain/tmp/pc_names.txt";
	    // read file and loop over machines
	    List<String> content=Files.readAllLines(Paths.get(fileName));
		// loop over all machines in the list
		for (String line:content) {
	    	// command to test connection
	    	String command1 = "ssh rlegrand@" + line +" 'hostname'";
	    	// declare new Clean
	    	Clean clean = new Clean(command1, true, 3, "machine " + line + " is responding.", "machine " + line + " does not respond.");
	    	// run process
	    	boolean success1 = clean.runAndCheck();
	    	// print the outcome of the test
	    	clean.printOutput(success1);
	    	// if the machine is responding, go on: suppress the /tmp/rlegrand directory
	    	if (success1) {
	    		// command to suppress directory
	    		String command2 = "ssh rlegrand@" + line + " rm -R /cal/homes/rlegrand/tmp/rlegrand";
	    		// declare new Clean
	    		Clean clean2 = new Clean(command2, true, 10, "directory suppressed on " + line + ".", "failed to suppress directory on " + line + ".");
	    		// run process
		    	boolean success2 = clean2.runAndCheck();
		    	// print the outcome of the test
		    	clean2.printOutput(success2);
	    	}
		}
	}
	
	
	// main method
	public static void main(String[] args) throws InterruptedException, IOException {
			
		// step 8.1
		clean();
	}		
	
}
