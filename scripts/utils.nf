@Grab('com.xlson.groovycsv:groovycsv:1.3')
import static com.xlson.groovycsv.CsvParser.parseCsv

def getMeanFD( csv ) {
    /*
    Compute meanFD given a CSV file
    */
    def sum = 0;
    def lines = 0;
    for (line in parseCsv(new FileReader(csv.toString()), separator: "\t")){
        try {
            sum += Double.parseDouble(line.framewise_displacement)
            lines++;
        } catch(NumberFormatException e) {
            continue 
        }
    }
    sum/lines 

}
