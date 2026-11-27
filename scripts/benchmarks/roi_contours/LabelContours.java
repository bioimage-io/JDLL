import java.util.Arrays;

/** Benchmark kernel. Only the JDK is used. Inputs are validated integer-valued label planes. */
final class LabelContours {
    // Directions proceed clockwise on an image with Y increasing downwards.
    private static final int[] DX = {1, 0, -1, 0};
    private static final int[] DY = {0, 1, 0, -1};
    private static final int[] RIGHT_X = {0, -1, -1, 0};
    private static final int[] RIGHT_Y = {0, 0, -1, -1};
    private static final int[] LEFT_X = {0, 0, -1, -1};
    private static final int[] LEFT_Y = {-1, 0, 0, -1};

    /** Triples of label ID, linear start, length. Runs never wrap to the next row. */
    static int[] runs(float[] pixels, int width) {
        int[] result = new int[384];
        int size = 0;
        for (int row = 0; row < pixels.length; row += width) {
            int end = row + width;
            int p = row;
            while (p < end) {
                int start = p++;
                float id = pixels[start];
                while (p < end && pixels[p] == id) p++;
                if (id == 0) continue;
                if (size == result.length) result = Arrays.copyOf(result, size * 2);
                result[size++] = (int) id;
                result[size++] = start;
                result[size++] = p - start;
            }
        }
        return Arrays.copyOf(result, size);
    }

    private static boolean member(float[] pixels, int w, int h, int x, int y, int id) {
        return x >= 0 && x < w && y >= 0 && y < h && pixels[y * w + x] == id;
    }

    /** Repeated records [label ID, corner count, x0, y0, ...]. Rings close implicitly. */
    static int[] rings(float[] pixels, int width, int[] runs) {
        int height = pixels.length / width;
        boolean[] seenEastSide = new boolean[pixels.length];
        // Two vertical edges per run, at most two corners per vertical edge and one ring per run.
        int[] result = new int[Math.multiplyExact(runs.length / 3, 10)];
        int size = 0;
        for (int i = 0; i < runs.length; i += 3) {
            int id = runs[i], eastPixel = runs[i + 1] + runs[i + 2] - 1;
            if (seenEastSide[eastPixel]) continue;
            int startX = eastPixel % width + 1, startY = eastPixel / width;
            int x = startX, y = startY, direction = 1;
            result[size++] = id;
            int countIndex = size++;
            do {
                if (direction == 1) seenEastSide[y * width + x - 1] = true;
                x += DX[direction];
                y += DY[direction];
                int next = direction;
                if (!member(pixels, width, height, x + RIGHT_X[direction], y + RIGHT_Y[direction], id)) {
                    next = (direction + 1) & 3;
                } else if (member(pixels, width, height, x + LEFT_X[direction], y + LEFT_Y[direction], id)) {
                    next = (direction + 3) & 3;
                }
                if (direction != next) {
                    result[size++] = x;
                    result[size++] = y;
                }
                direction = next;
            } while (x != startX || y != startY || direction != 1);
            result[countIndex] = (size - countIndex - 1) / 2;
        }
        return Arrays.copyOf(result, size);
    }
}
