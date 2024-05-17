using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MaterialSkin;
using MaterialSkin.Controls;
using static PIA_PROIMA.Filters;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.Util;
using Emgu.CV.Structure;
using System.Windows.Forms.DataVisualization.Charting;

namespace PIA_PROIMA
{
    public partial class MainForm : MaterialForm
    {
        VideoCapture webcam;
        CascadeClassifier haar;

        VideoCapture grabber;
        Image<Bgr, byte> currentFrame, currentFrameFiltered;
        double TotalDuration;
        double FrameCount;
        bool Video = false;
        string filterName = "";
        Mat m;
        private bool isPaused = false;
        private bool validationShown = false;

        readonly MaterialSkin.MaterialSkinManager materialSkinManager;

        public MainForm()
        {
            InitializeComponent();
            materialSkinManager = MaterialSkin.MaterialSkinManager.Instance;
            materialSkinManager.EnforceBackcolorOnAllComponents = true;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkin.MaterialSkinManager.Themes.LIGHT;
            materialSkinManager.ColorScheme = new MaterialSkin.ColorScheme(
                MaterialSkin.Primary.Blue500,
                MaterialSkin.Primary.Blue700,
                MaterialSkin.Primary.Blue100,
                MaterialSkin.Accent.Blue200,
                MaterialSkin.TextShade.WHITE
            );
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            webcam = new VideoCapture(0);
            haar = new CascadeClassifier("haarcascade_frontalface_default.xml");
        }

        private void tab_Exit_Paint(object sender, PaintEventArgs e)
        {
            Application.Exit();
        }

        private void btnLoadImage_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Archivos de imagen|*.jpg;*.jpeg;*.png;*.bmp|Todos los archivos|*.*";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    string imagePath = openFileDialog.FileName;
                    pbOriginalImage.Image = Image.FromFile(imagePath);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error al cargar la imagen: " + ex.Message);
                }
            }

            cbFilters.SelectedItem = null;
        }

        private void cbFilters_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (pbOriginalImage.Image == null)
            {
                MessageBox.Show("No hay ninguna imagen cargada a la cual aplicarle un filtro", "Error", MessageBoxButtons.OK);
                return;
            }
            if(cbFilters.SelectedItem != null)
            {
                string selectedFilter = cbFilters.SelectedItem.ToString();

                Bitmap originalImage = new Bitmap(pbOriginalImage.Image);

                Bitmap filteredImage = null;

                switch (selectedFilter)
                {
                    case "Gaussian Blur":
                        filteredImage = Filters.ApplyGaussianBlur(originalImage, 20);
                        break;
                    case "Pixelate":
                        filteredImage = Filters.ApplyPixelation(originalImage, 20);
                        break;
                    case "Fish Eye":
                        filteredImage = Filters.ApplyFishEyeEffect(originalImage, 5);
                        break;
                    case "Cold":
                        filteredImage = Filters.ApplyColdFilter(originalImage);
                        break;
                    case "Edge Detection":
                        filteredImage = Filters.ApplyEdgeDetection(originalImage);
                        break;
                    case "Vignette":
                        filteredImage = Filters.ApplyVignette(originalImage, 3);
                        break;
                    case "Brightness":
                        filteredImage = Filters.Brightness(originalImage, 50);
                        break; 
                    case "Contrast":
                        filteredImage = Filters.Contrast(originalImage, 30);
                        break; 
                    case "Saturation":
                        filteredImage = Filters.Saturation(originalImage, 0.2f);
                        break;
                    case "Comic":
                        filteredImage = Filters.Comic(originalImage);
                        break;
                    default:
                        break;

                }
                // Muestra la imagen filtrada en el PictureBox
                pbFilteredImage.Image = filteredImage;

                var optimizedImage = OptimizeBitmap(filteredImage, 10);

                getRGBChart(optimizedImage, chartRed, graphicImageContainer);
            }

        }

        private void btSaveImage_Click(object sender, EventArgs e)
        {
            Bitmap filteredImage = pbFilteredImage.Image as Bitmap;
            if (filteredImage != null)
            {
                SaveFileDialog saveFileDialog = new SaveFileDialog();
                saveFileDialog.Filter = "Archivos de imagen|*.png;*.jpg;*.jpeg;*.bmp|Todos los archivos|*.*";

                if (saveFileDialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        // Guarda la imagen filtrada en la ubicación especificada
                        filteredImage.Save(saveFileDialog.FileName, ImageFormat.Png);
                        MessageBox.Show("Imagen guardada correctamente.");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("Error al guardar la imagen: " + ex.Message);
                    }
                }
            }
            else
            {
                MessageBox.Show("No hay una imagen filtrada para guardar.");
            }
        }

        private void btnActivarCamara_Click(object sender, EventArgs e)
        {
            timer1.Start();
        }

        private void getRGBChart(Bitmap image, Chart chart, FlowLayoutPanel container)
        {
            container.Visible = true;

            // Limpiar cualquier configuración previa
            chart.Series.Clear();
            chart.ChartAreas.Clear();

            // Crear un área de gráfico
            ChartArea chartArea = new ChartArea();
            chart.ChartAreas.Add(chartArea);

            // Crear una serie para cada canal de color
            Series redSeries = new Series("Red");
            redSeries.Color = Color.Red;
            Series greenSeries = new Series("Green");
            greenSeries.Color = Color.Green;
            Series blueSeries = new Series("Blue");
            blueSeries.Color = Color.Blue;

            // Lock the bits of the image
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, image.PixelFormat);
            int bytesPerPixel = Bitmap.GetPixelFormatSize(image.PixelFormat) / 8;

            unsafe
            {
                byte* ptr = (byte*)imageData.Scan0;

                // Recorrer la imagen y agregar los valores de cada canal de color a las series correspondientes
                for (int y = 0; y < image.Height; y++)
                {
                    for (int x = 0; x < image.Width; x++)
                    {
                        int offset = (y * imageData.Stride) + (x * bytesPerPixel);

                        // Obtener los valores de cada canal de color
                        int redValue = ptr[offset + 2]; // Red
                        int greenValue = ptr[offset + 1]; // Green
                        int blueValue = ptr[offset]; // Blue

                        // Agregar los valores a las series correspondientes
                        redSeries.Points.AddXY(x * image.Height + y, redValue);
                        greenSeries.Points.AddXY(x * image.Height + y, greenValue);
                        blueSeries.Points.AddXY(x * image.Height + y, blueValue);
                    }
                }
            }

            // Unlock the bits of the image
            image.UnlockBits(imageData);

            // Agregar las series al gráfico
            chart.Series.Add(redSeries);
            chart.Series.Add(greenSeries);
            chart.Series.Add(blueSeries);   
        }

        public static Bitmap OptimizeBitmap(Bitmap originalBitmap, int sampleRate)
        {
            // Obtener las dimensiones de la imagen original
            int originalWidth = originalBitmap.Width;
            int originalHeight = originalBitmap.Height;

            // Calcular las nuevas dimensiones después del muestreo
            int newWidth = originalWidth / sampleRate;
            int newHeight = originalHeight / sampleRate;

            // Crear un nuevo bitmap para la imagen optimizada
            Bitmap optimizedBitmap = new Bitmap(newWidth, newHeight);

            // Iterar a través de los píxeles del bitmap original y copiar los valores muestreados al bitmap optimizado
            for (int x = 0; x < newWidth; x++)
            {
                for (int y = 0; y < newHeight; y++)
                {
                    // Obtener el color del píxel muestreado
                    Color pixelColor = originalBitmap.GetPixel(x * sampleRate, y * sampleRate);

                    // Establecer el color del píxel en el bitmap optimizado
                    optimizedBitmap.SetPixel(x, y, pixelColor);
                }
            }

            return optimizedBitmap;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            int faceCounter = 0;
            using (Image<Bgr, byte> binaryGoruntu = webcam.QueryFrame().ToImage<Bgr, byte>())
            {
                if (binaryGoruntu != null)
                {
                    Image<Gray, byte> greyScaleGoruntu = binaryGoruntu.Convert<Gray, byte>();
                    Rectangle[] rectangles = haar.DetectMultiScale(greyScaleGoruntu, 1.4, 1, new Size(100, 100), new Size(800, 800));

                    // Lista de colores predefinidos
                    Bgr[] colors = { new Bgr(255, 0, 0), // Rojo
                             new Bgr(0, 255, 0), // Verde
                             new Bgr(0, 0, 255), // Azul
                             new Bgr(255, 255, 0), // Amarillo
                             new Bgr(255, 0, 255), // Magenta
                             new Bgr(0, 255, 255) }; // Cyan

                    for (int i = 0; i < rectangles.Length; i++)
                    {
                        // Asignamos el color correspondiente al índice del rectángulo
                        Bgr color = colors[i % colors.Length];
                        binaryGoruntu.Draw(rectangles[i], color, 3);
                        faceCounter++;
                    }
                    pbVideoSource.Image = binaryGoruntu.ToBitmap();
                    txtFaceCounter.Text = faceCounter.ToString();
                }
            }
        }

        private void btnLoadVideo_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Archivos de imagen|*.mp4;*.webm;*.avi;*.mov|Todos los archivos|*.*";

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    grabber = new VideoCapture(openFileDialog.FileName);
                    m = new Mat();
                    grabber.Read(m);
                    currentFrame = m.ToImage<Bgr, byte>();

                    pbOriginalImage.Image = currentFrame.Bitmap;
                    TotalDuration = grabber.GetCaptureProperty(CapProp.FrameCount);
                    FrameCount = grabber.GetCaptureProperty(CapProp.PosFrames);
                    Video = true;

                    tbVideo.Maximum = (int)TotalDuration;


                    //string videoPath = openFileDialog.FileName;
                    //wmpOriginal.URL = videoPath;
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error al cargar la imagen: " + ex.Message);
                }
            }

            cbVideoFilters.SelectedItem = null;

        }

        private void btnPlayPause_Click(object sender, EventArgs e)
        {
            if (Video)
            {
                isPaused = !isPaused; // Invertir el estado de isPaused

                if (isPaused)
                {
                    // Si está en pausa, reanudamos la reproducción
                    Application.Idle += new EventHandler(VideoFrameCapture);
                }
                else
                {
                    // Si no está en pausa, pausamos la reproducción
                    Application.Idle -= new EventHandler(VideoFrameCapture);
                }
            }
            else
            {
                MessageBox.Show("Video no cargado", "Error", MessageBoxButtons.OK);
            }
        }

        private void VideoFrameCapture(object sender, EventArgs e)
        {
            if (FrameCount < TotalDuration - 2)
            {
                m = grabber.QueryFrame();
                currentFrame = m.ToImage<Bgr, byte>();
                currentFrameFiltered = m.ToImage<Bgr, byte>();

                if (cbVideoFilters.SelectedItem != null)
                {
                    string selectedFilter = cbVideoFilters.SelectedItem.ToString();
                    currentFrameFiltered = ApplyFilter(currentFrameFiltered, selectedFilter);
                }

                FrameCount = grabber.GetCaptureProperty(CapProp.PosFrames);

                // Actualizar la posición del TrackBar
                tbVideo.Value = (int)FrameCount;
            }
            else
            {
                FrameCount = 0;
                grabber.SetCaptureProperty(CapProp.PosFrames, 0);
            }

            pbOriginalVideo.Image = currentFrame.Bitmap;
            pbFilteredVideo.Image = currentFrameFiltered.Bitmap;

            var optimizedImage = OptimizeBitmap(currentFrameFiltered.Bitmap, 10);
            getRGBChart(optimizedImage, chartRedVideo, flowLayoutPanel14);
        }

        private void tbVideo_Scroll(object sender, EventArgs e)
        {
            if (Video && grabber != null)
            {
                double totalFrames = TotalDuration;
                double currentPosition = (tbVideo.Value / (double)tbVideo.Maximum) * totalFrames;
                grabber.SetCaptureProperty(CapProp.PosFrames, currentPosition);
            }
        }

        private void cbVideoFilters_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (!Video && grabber == null)
            {
                MessageBox.Show("No hay ningun video cargado a la cual aplicarle un filtro", "Error", MessageBoxButtons.OK);
                return;
            }
            if (cbFilters.SelectedItem == null)
            {
                return;
            }
        }

        private Image<Bgr, byte> ApplyFilter(Image<Bgr, byte> frame, string filterName)
        {
            Bitmap bitmapFrame = frame.Bitmap;

            switch (filterName)
            {
                case "Gaussian Blur":
                    return new Image<Bgr, byte>(Filters.ApplyGaussianBlur(bitmapFrame, 20));
                case "Pixelate":
                    return new Image<Bgr, byte>(Filters.ApplyPixelation(bitmapFrame, 20));
                case "Fish Eye":
                    return new Image<Bgr, byte>(Filters.ApplyFishEyeEffect(bitmapFrame, 5));
                case "Cold":
                    return new Image<Bgr, byte>(Filters.ApplyColdFilter(bitmapFrame));
                case "Edge Detection":
                    return new Image<Bgr, byte>(Filters.ApplyEdgeDetection(bitmapFrame));
                case "Vignette":
                    return new Image<Bgr, byte>(Filters.ApplyVignette(bitmapFrame, 3));
                case "Brightness":
                    return new Image<Bgr, byte>(Filters.Brightness(bitmapFrame, 50)); 
                case "Contrast":
                    return new Image<Bgr, byte>(Filters.Contrast(bitmapFrame, 30));
                case "Saturation":
                    return new Image<Bgr, byte>(Filters.Saturation(bitmapFrame, 0.2f));
                case "Comic":
                    return new Image<Bgr, byte>(Filters.Comic(bitmapFrame));
                default:
                    return frame;
            }
        }
    }
}
