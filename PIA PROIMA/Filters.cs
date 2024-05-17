using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace PIA_PROIMA
{
    class Filters
    {
        public static Bitmap ApplyGaussianBlur(Bitmap image, int radius)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            // Convertir la imagen a un formato compatible para procesamiento
            BitmapData sourceData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                                   ImageLockMode.ReadOnly,
                                                   PixelFormat.Format32bppArgb);

            BitmapData resultData = result.LockBits(new Rectangle(0, 0, image.Width, image.Height),
                                                     ImageLockMode.WriteOnly,
                                                     PixelFormat.Format32bppArgb);

            int bytesPerPixel = 4;

            // Determinar el tamaño de la matriz de convolución
            int kernelSize = radius * 2 + 1;
            float[,] kernel = new float[kernelSize, kernelSize];
            float sigma = radius / 3.0f;
            float sigma22 = 2 * sigma * sigma;
            float sqrtPiSigma22 = (float)Math.Sqrt(Math.PI * sigma22);
            float radius2 = radius * radius;
            float total = 0;

            // Llenar la matriz de convolución con valores de un filtro gaussiano y calcular el total
            for (int y = -radius; y <= radius; y++)
            {
                for (int x = -radius; x <= radius; x++)
                {
                    float distance = (x * x + y * y);
                    if (distance > radius2)
                        kernel[y + radius, x + radius] = 0;
                    else
                    {
                        float value = (float)Math.Exp(-distance / sigma22) / sqrtPiSigma22;
                        kernel[y + radius, x + radius] = value;
                        total += value;
                    }
                }
            }

            // Normalizar el kernel
            if (total != 0)
            {
                for (int y = 0; y < kernelSize; y++)
                {
                    for (int x = 0; x < kernelSize; x++)
                    {
                        kernel[y, x] /= total;
                    }
                }
            }

            // Aplicar el filtro
            unsafe
            {
                byte* sourcePointer = (byte*)sourceData.Scan0;
                byte* resultPointer = (byte*)resultData.Scan0;

                int sourceStride = sourceData.Stride;
                int resultStride = resultData.Stride;

                int sourceOffset = sourceStride - image.Width * bytesPerPixel;
                int resultOffset = resultStride - image.Width * bytesPerPixel;

                for (int y = radius; y < image.Height - radius; y++)
                {
                    for (int x = radius; x < image.Width - radius; x++)
                    {
                        float blue = 0;
                        float green = 0;
                        float red = 0;

                        for (int ky = -radius; ky <= radius; ky++)
                        {
                            for (int kx = -radius; kx <= radius; kx++)
                            {
                                int currentY = y + ky;
                                int currentX = x + kx;
                                byte* currentPixel = sourcePointer + currentY * sourceStride + currentX * bytesPerPixel;
                                float weight = kernel[ky + radius, kx + radius];

                                blue += currentPixel[0] * weight;
                                green += currentPixel[1] * weight;
                                red += currentPixel[2] * weight;
                            }
                        }

                        byte* resultPixel = resultPointer + y * resultStride + x * bytesPerPixel;
                        resultPixel[0] = (byte)Math.Max(0, Math.Min(255, blue));
                        resultPixel[1] = (byte)Math.Max(0, Math.Min(255, green));
                        resultPixel[2] = (byte)Math.Max(0, Math.Min(255, red));
                        resultPixel[3] = 255; // Alpha
                    }
                }
            }

            // Liberar los recursos de las imágenes
            image.UnlockBits(sourceData);
            result.UnlockBits(resultData);

            return result;
        }
        public static Bitmap ApplyPixelation(Bitmap image, int pixelSize)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            // Recorre la imagen en bloques del tamaño de pixelSize
            for (int y = 0; y < image.Height; y += pixelSize)
            {
                for (int x = 0; x < image.Width; x += pixelSize)
                {
                    // Obtén el color promedio del bloque actual
                    Color avgColor = GetAverageColor(image, x, y, pixelSize);

                    // Rellena el bloque actual con el color promedio
                    for (int py = y; py < y + pixelSize && py < image.Height; py++)
                    {
                        for (int px = x; px < x + pixelSize && px < image.Width; px++)
                        {
                            result.SetPixel(px, py, avgColor);
                        }
                    }
                }
            }

            return result;
        }

        private static Color GetAverageColor(Bitmap image, int startX, int startY, int blockSize)
        {
            int totalR = 0, totalG = 0, totalB = 0;

            // Recorre el bloque de píxeles y suma los valores de color
            for (int y = startY; y < startY + blockSize && y < image.Height; y++)
            {
                for (int x = startX; x < startX + blockSize && x < image.Width; x++)
                {
                    Color pixelColor = image.GetPixel(x, y);
                    totalR += pixelColor.R;
                    totalG += pixelColor.G;
                    totalB += pixelColor.B;
                }
            }

            // Calcula el promedio de los valores de color
            int pixelCount = blockSize * blockSize;
            int avgR = totalR / pixelCount;
            int avgG = totalG / pixelCount;
            int avgB = totalB / pixelCount;

            return Color.FromArgb(avgR, avgG, avgB);
        }
        public static Bitmap ApplyFishEyeEffect(Bitmap image, int strength)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            int width = image.Width;
            int height = image.Height;
            int radius = Math.Min(width, height) / strength;

            double centerX = width / 2.0;
            double centerY = height / 2.0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double dx = x - centerX;
                    double dy = y - centerY;
                    double distance = Math.Sqrt(dx * dx + dy * dy);

                    if (distance < radius)
                    {
                        double r = distance / radius;
                        double angle = Math.Atan2(dy, dx);
                        double nr = Math.Pow(r, 2.0 / 3.0);

                        int newX = (int)(centerX + nr * radius * Math.Cos(angle));
                        int newY = (int)(centerY + nr * radius * Math.Sin(angle));

                        if (newX >= 0 && newX < width && newY >= 0 && newY < height)
                        {
                            result.SetPixel(x, y, image.GetPixel(newX, newY));
                        }
                    }
                }
            }

            return result;
        }
        public static Bitmap ApplyColdFilter(Bitmap image)
        {
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, image.PixelFormat);
            int bytesPerPixel = Bitmap.GetPixelFormatSize(image.PixelFormat) / 8;
            int byteCount = imageData.Stride * image.Height;
            byte[] pixels = new byte[byteCount];
            Marshal.Copy(imageData.Scan0, pixels, 0, pixels.Length);

            // Ajusta el tono azul en cada píxel
            for (int i = 0; i < byteCount; i += bytesPerPixel)
            {
                int blue = pixels[i];
                int green = pixels[i + 1];
                int red = pixels[i + 2];

                // Aplica un tono azul
                blue = (int)Math.Min(255, blue * 1.5);

                // Asigna los nuevos valores de color
                pixels[i] = (byte)blue;
                pixels[i + 1] = (byte)green;
                pixels[i + 2] = (byte)red;
            }

            Marshal.Copy(pixels, 0, imageData.Scan0, pixels.Length);
            image.UnlockBits(imageData);

            return image;
        }
        public static Bitmap ApplyEdgeDetection(Bitmap image)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            int[,] sobelX = {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
            };

            int[,] sobelY = {
                {-1, -2, -1},
                {0, 0, 0},
                {1, 2, 1}
            };

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    int pixelX = 0, pixelY = 0;

                    // Aplicar el filtro de Sobel en dirección X
                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            Color pixel = image.GetPixel(x + j, y + i);
                            int gray = (int)(pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114);
                            pixelX += gray * sobelX[i + 1, j + 1];
                        }
                    }

                    // Aplicar el filtro de Sobel en dirección Y
                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            Color pixel = image.GetPixel(x + j, y + i);
                            int gray = (int)(pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114);
                            pixelY += gray * sobelY[i + 1, j + 1];
                        }
                    }

                    int magnitude = (int)Math.Sqrt(pixelX * pixelX + pixelY * pixelY);
                    magnitude = Math.Min(255, magnitude);
                    result.SetPixel(x, y, Color.FromArgb(magnitude, magnitude, magnitude));
                }
            }

            return result;
        }
        public static Bitmap ApplyVignette(Bitmap image, int strength)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            double centerX = image.Width / 2.0;
            double centerY = image.Height / 2.0;
            double maxDistance = Math.Sqrt(centerX * centerX + centerY * centerY);

            // Pre-calcular los valores de alpha para cada posición
            byte[,] alphas = new byte[image.Width, image.Height];
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    double distance = Math.Sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                    double ratio = distance / maxDistance;
                    int alpha = (int)(255 - 255 * Math.Pow(ratio, strength));
                    alphas[x, y] = (byte)Math.Max(0, Math.Min(255, alpha));
                }
            }

            // Aplicar los valores de alpha pre-calculados a la imagen resultante
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color originalColor = image.GetPixel(x, y);
                    Color newColor = Color.FromArgb(alphas[x, y], originalColor.R, originalColor.G, originalColor.B);
                    result.SetPixel(x, y, newColor);
                }
            }

            return result;
        }
        public static Bitmap Brightness(Bitmap image, int brightness)
        {
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, image.PixelFormat);

            int bytesPerPixel = Bitmap.GetPixelFormatSize(image.PixelFormat) / 8;
            int byteCount = bmpData.Stride * image.Height;
            byte[] pixels = new byte[byteCount];
            Marshal.Copy(bmpData.Scan0, pixels, 0, byteCount);

            for (int i = 0; i < byteCount; i += bytesPerPixel)
            {
                int blue = pixels[i];
                int green = pixels[i + 1];
                int red = pixels[i + 2];

                // Aumentar el brillo sumando el valor especificado
                blue = Clamp(blue + brightness, 0, 255);
                green = Clamp(green + brightness, 0, 255);
                red = Clamp(red + brightness, 0, 255);

                pixels[i] = (byte)blue;
                pixels[i + 1] = (byte)green;
                pixels[i + 2] = (byte)red;
            }

            Marshal.Copy(pixels, 0, bmpData.Scan0, byteCount);
            image.UnlockBits(bmpData);

            return image;
        }

        public static Bitmap Contrast(Bitmap image, float contrast)
        {
            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, image.PixelFormat);

            int bytesPerPixel = Bitmap.GetPixelFormatSize(image.PixelFormat) / 8;
            int byteCount = bmpData.Stride * image.Height;
            byte[] pixels = new byte[byteCount];
            Marshal.Copy(bmpData.Scan0, pixels, 0, byteCount);

            float contrastFactor = (100f + contrast) / 100f;
            contrastFactor *= contrastFactor;

            for (int i = 0; i < byteCount; i += bytesPerPixel)
            {
                float blue = pixels[i];
                float green = pixels[i + 1];
                float red = pixels[i + 2];

                // Aplicar transformación lineal para aumentar el contraste
                blue = ((blue / 255f - 0.5f) * contrastFactor + 0.5f) * 255f;
                green = ((green / 255f - 0.5f) * contrastFactor + 0.5f) * 255f;
                red = ((red / 255f - 0.5f) * contrastFactor + 0.5f) * 255f;

                pixels[i] = (byte)Clamp(blue, 0, 255);
                pixels[i + 1] = (byte)Clamp(green, 0, 255);
                pixels[i + 2] = (byte)Clamp(red, 0, 255);
            }

            Marshal.Copy(pixels, 0, bmpData.Scan0, byteCount);
            image.UnlockBits(bmpData);

            return image;
        }
        // Función auxiliar para limitar un valor entre un mínimo y un máximo
        private static float Clamp(float value, float min, float max)
        {
            return Math.Max(min, Math.Min(max, value));
        }
        // Función auxiliar para limitar un valor entre un mínimo y un máximo
        private static int Clamp(int value, int min, int max)
        {
            return Math.Max(min, Math.Min(max, value));
        }

        public static Bitmap Saturation(Bitmap image, float saturationFactor)
        {
            Bitmap result = new Bitmap(image.Width, image.Height);

            // Iterar sobre cada píxel de la imagen
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Color originalColor = image.GetPixel(x, y);

                    // Convertir el color original al espacio de color HSV
                    float hue, saturation, value;
                    ColorToHSV(originalColor, out hue, out saturation, out value);

                    // Reducir la saturación multiplicando por el factor de saturación
                    saturation *= saturationFactor;

                    // Convertir el color de nuevo a RGB
                    Color newColor = HSVToColor(hue, saturation, value);

                    // Asignar el nuevo color al píxel en la imagen resultante
                    result.SetPixel(x, y, newColor);
                }
            }

            return result;
        }

        // Función auxiliar para convertir un color RGB a HSV
        private static void ColorToHSV(Color color, out float hue, out float saturation, out float value)
        {
            int max = Math.Max(color.R, Math.Max(color.G, color.B));
            int min = Math.Min(color.R, Math.Min(color.G, color.B));
            float delta = max - min;

            hue = GetHue(color, max, delta);
            saturation = (max == 0) ? 0 : 1f - (1f * min / max);
            value = max / 255f;
        }

        // Función auxiliar para obtener el componente de tono (Hue) en el espacio de color HSV
        private static float GetHue(Color color, int max, float delta)
        {
            if (delta == 0)
                return 0;

            float hue;
            if (color.R == max)
                hue = (color.G - color.B) / delta;
            else if (color.G == max)
                hue = 2f + (color.B - color.R) / delta;
            else
                hue = 4f + (color.R - color.G) / delta;

            hue *= 60;
            if (hue < 0)
                hue += 360;

            return hue;
        }

        // Función auxiliar para convertir un color HSV a RGB
        private static Color HSVToColor(float hue, float saturation, float value)
        {
            int hi = (int)(Math.Floor(hue / 60) % 6);
            float f = hue / 60 - (float)Math.Floor(hue / 60);

            float v = value * 255;
            int p = (int)(value * (1 - saturation) * 255);
            int q = (int)(value * (1 - f * saturation) * 255);
            int t = (int)(value * (1 - (1 - f) * saturation) * 255);

            if (hi == 0)
                return Color.FromArgb(255, (int)v, t, p);
            else if (hi == 1)
                return Color.FromArgb(255, q, (int)v, p);
            else if (hi == 2)
                return Color.FromArgb(255, p, (int)v, t);
            else if (hi == 3)
                return Color.FromArgb(255, p, q, (int)v);
            else if (hi == 4)
                return Color.FromArgb(255, t, p, (int)v);
            else
                return Color.FromArgb(255, (int)v, p, q);
        }
        public static Bitmap Comic(Bitmap image)
        {
            // Convertir la imagen a escala de grises
            Image<Gray, byte> grayImage = new Image<Gray, byte>(image);

            // Aplicar un filtro de borde (detectar contornos)
            Image<Gray, byte> edgesImage = grayImage.Canny(100, 200);

            // Realzar el contraste
            edgesImage._EqualizeHist();

            // Aplicar una binarización
            edgesImage._ThresholdBinaryInv(new Gray(50), new Gray(255));

            // Convertir la imagen resultante a formato Bitmap
            Bitmap resultBitmap = edgesImage.ToBitmap();

            return resultBitmap;
        }
    }
}

