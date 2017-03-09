import java.io.File;
import java.net.InetAddress;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * To include srcAddress, change lines 42 and 96.
 * @author macandcheese
 */
public class TraceRouteParse {
    
    
    public static void main(String[] args) {
        
        try {
            Scanner s = new Scanner(new File("topo-v6.l8.20161101.1477985100.txt"));
            String line;
            String subline;
            String srcAddr = "";
            String destAddr;
            String routeAddr;
            int hops;
            
            InetAddress srcInet;
            String srcBits = "";
            String srcArray[];
            byte srcNibble[] = new byte[8];
            
            Matrix features = new Matrix(); // 64 bits of homeAddr and 64 bits of destAddr
            Matrix labels = new Matrix();   // Number of hops
            features.newColumns(64);       // Initialized to 0
            labels.newColumn();             // **We could add another column for time (ms)
            
            // Check map, each address will be stored in string format 
            // To check for previous existing entries to prevent duplicates
            HashMap<String, AtomicInteger> checkLog = new HashMap<>();
            
            // Regex credit to https://gist.github.com/syzdek/6086792
            String IPV4SEG  = "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])";
            String IPV4ADDR = "("+IPV4SEG+"\\.){3,3}"+IPV4SEG;
            String IPV6SEG  = "[0-9a-fA-F]{1,4}";
            String IPV6ADDR = "("
                    + "("+IPV6SEG+":){7,7}"+IPV6SEG+"|"            // 1:2:3:4:5:6:7:8"
                    + "("+IPV6SEG+":){1,7}:|"                      // 1::                                 1:2:3:4:5:6:7::"
                    + "("+IPV6SEG+":){1,6}:"+IPV6SEG+"|"           // 1::8               1:2:3:4:5:6::8   1:2:3:4:5:6::8
                    + "("+IPV6SEG+":){1,5}(:"+IPV6SEG+"){1,2}|"    // 1::7:8             1:2:3:4:5::7:8   1:2:3:4:5::8
                    + "("+IPV6SEG+":){1,4}(:"+IPV6SEG+"){1,3}|"    // 1::6:7:8           1:2:3:4::6:7:8   1:2:3:4::8"
                    + "("+IPV6SEG+":){1,3}(:"+IPV6SEG+"){1,4}|"    // 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8"
                    + "("+IPV6SEG+":){1,2}(:"+IPV6SEG+"){1,5}|"    // 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8"
                    + ""+IPV6SEG+":((:"+IPV6SEG+"){1,6})|"         // 1::3:4:5:6:7:8     1::3:4:5:6:7:8   1::8"
                    + ":((:"+IPV6SEG+"){1,7}|:)|"                  // ::2:3:4:5:6:7:8    ::2:3:4:5:6:7:8  ::8       ::       "
                    + "fe80:(:"+IPV6SEG+"){0,4}%[0-9a-zA-Z]{1,}|"  // fe80::7:8%eth0     fe80::7:8%1  (link-local IPv6 addresses with zone index)"
                    + "::(ffff(:0{1,4}){0,1}:){0,1}"+IPV4ADDR+"|"  // ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)"
                    + "("+IPV6SEG+":){1,4}:"+IPV4ADDR               // 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33 (IPv4-Embedded IPv6 Address)"
                    + ")";
            
            Pattern r = Pattern.compile(IPV6ADDR);
            Matcher m;
            
            while(s.hasNextLine()) {
                line = s.nextLine();
                
                // If - Check to see if the read line is a hop count line
                if(Character.isDigit(line.charAt(1)) || Character.isDigit(line.charAt(0))) {
                    
                    // Disregard any entries without data
                    if(line.charAt(4) == '*') {
                        continue;
                    }
                    
                    // Remove the hop count from the string
                    hops = Integer.parseInt(line.substring(0, 2).trim());
                    m = r.matcher(line);
                    
                    if(m.find()) {
                        // ------ Grab the IPv6 Address from the line
                        routeAddr = m.group(0);
                        //System.out.println(routeAddr);
                        InetAddress a = InetAddress.getByName(routeAddr);
                        String arr = Arrays.toString(a.getAddress());
                        String destArray[] = arr.split(", ");
                        destArray[0] = destArray[0].substring(1);
                        //System.out.println(array[0] + " " + array[1] + " " + array[2] + " " + array[3]);
                        
                        // Parse and reformat the IPv6 address from String to Bytes to the string representation of the bits
                        byte nibble[] = new byte[8];
                        String destBits = "";
                        for(int i = 0; i < nibble.length; i++) {
                            nibble[i] = (byte) Integer.parseInt(destArray[i]);
                            destBits += String.format("%8s", Integer.toBinaryString(nibble[i] & 0xFF)).replace(' ', '0');
                        }
                        
                        // Check to see if address is already in Matrix. 
                        // If yes, break and continue to next entry. 
                        // Else, add entry to features Matrix.
                        if(checkLog.containsKey(destBits)) {
                            // Add 1 to existing value and continue to next entry
                            checkLog.get(destBits).incrementAndGet();
                            continue;
                        }
                        else {
                            checkLog.put(destBits, new AtomicInteger(1));
                        }
                        
                        
                        // Prepare the matrix by adding another row
                        features.newRow();
                        
                        // ----- Split the string and place each bit into a column of the row for the matrix
                        //String rowEntry = srcBits + destBits;     // This is removed because srcBits is the same for all
                        String rowEntry = destBits;
                        String[] splitEntry = rowEntry.split("");
                        for(int i = 0; i < features.cols(); i++) {
                            features.row(features.rows()-1)[i] = Double.parseDouble(splitEntry[i]);
                        }
                        
                        // Add the hop count to the labels matrix
                        labels.newRow();
                        labels.row(labels.rows()-1)[0] = (double)hops;
                    }
                }
                // Else - Check to see if the read line is the start of the traceroute
                else {
                    m = r.matcher(line);
                    destAddr = line.substring(line.indexOf(" to ")+4); // Grabs destination address
                    if(m.find()) {
                        srcAddr = m.group(0); // Grabs start address
                        
                        srcInet = InetAddress.getByName(srcAddr);
                        srcBits = Arrays.toString(srcInet.getAddress());
                        srcArray = srcBits.split(", ");
                        srcArray[0] = srcArray[0].substring(1);
                        
                        srcBits = "";
                        for(int i = 0; i < 8; i++) {
                            srcNibble[i] = (byte) Integer.parseInt(srcArray[i]);
                            srcBits += String.format("%8s", Integer.toBinaryString(srcNibble[i] & 0xFF)).replace(' ', '0');
                        }
                    }
                }
            }
            
            // Print out contents of features
//            for(int i = 0; i < features.rows(); i++) {
//                System.out.print(i + ": ");
//                for(int j = 0; j < features.cols(); j++) {
//                    System.out.print(features.row(i)[j] + " ");
//                }
//                System.out.println();
//            }
            
            features.saveARFF("traceroute_features.arff");
            labels.saveARFF("traceroute_labels.arff");
        }
        catch(Exception e) {
            System.out.println("File not found.");
            
        }
        
    }
    
    // This is to handle multiple traceroute files in future
    public static void readAllFilesInDirectory() {
    File dir = new File("myDirectoryPath");
    File[] directoryListing = dir.listFiles();
    if (directoryListing != null) {
        for (File child : directoryListing) {
            // Do something with child
        }
    } else {
            // Handle the case where dir is not really a directory.
        }
    }
    
    
}
