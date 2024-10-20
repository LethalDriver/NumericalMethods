public class Main {
    public static void main(String[] args) {
        long startTime = System.nanoTime();
        double k = 1;
        double sum = 0;
        double n;

        while (true) {
            n = 1/Math.pow(k, 2);
            double prev = sum;
            sum += n;
            if (sum == prev) {
                break;
            }
            k += 1;
        }
        double analytical = Math.pow(Math.PI, 2)/6;
        double difference = (1-(sum/analytical))*100;

        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        double durationInSeconds = duration / 1_000_000.0;

        System.out.printf("Sum for n_max: %.10f", sum);
        System.out.println();
        System.out.printf("Difference: %.10f%%", difference);
        System.out.println();
        System.out.printf("Execution time: %.2f milliseconds", durationInSeconds);
    }
}