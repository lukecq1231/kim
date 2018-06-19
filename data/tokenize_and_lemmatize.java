import edu.stanford.nlp.simple.*;
import java.io.*;
import java.util.*;

public class tokenize_and_lemmatize {
    public static void main(String[] args) throws FileNotFoundException{ 
        // Open the file
        try{
            Scanner scan = new Scanner(new File(args[0]));
            PrintWriter writer1 = new PrintWriter(args[1], "UTF-8");
            PrintWriter writer2 = new PrintWriter(args[2], "UTF-8");

            while(scan.hasNextLine()){
                String line = scan.nextLine();
                // System.out.println(line);
                Sentence sent = new Sentence(line);
                List<String> words = sent.words();
                String joined1 = String.join(" ", words);
                List<String> lemmas = sent.lemmas();
                String joined2 = String.join(" ", lemmas);
                writer1.println(joined1);
                writer2.println(joined2);
            }
            writer1.close();
            writer2.close();

        } catch (Exception e) {
           e.printStackTrace();
        }
    }
}