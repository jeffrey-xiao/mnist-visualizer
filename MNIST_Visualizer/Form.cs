using Machine_Learning.Neural_Network;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MNIST_DEMO {
    public partial class Form1 : Form {
        private const int SIZE = 28;
        private static Network network;
        private static bool isMouseDown = false;
        private Graphics g;
        private double[,] intensity;

        public Form1 () {
            intensity = new double[SIZE, SIZE];
            InitializeComponent();
            this.Paint += initial_paint;
        }

        private void initial_paint (object sender, PaintEventArgs e) {
            e.Graphics.FillRectangle(Brushes.White, 10, 10, 28 * 8, 28 * 8);
        }

        private void loadWeightsBtn_Click (object sender, EventArgs e) {
            network = new Network(weightsPath.Text);
            this.Text = "Finish loading weights";
        }

        private void Form1_MouseDown (object sender, MouseEventArgs e) {
            g = this.CreateGraphics();
            isMouseDown = true;
            while (isMouseDown) {
                Application.DoEvents();
                Point p = PointToClient(Cursor.Position);
                p.X = (p.X - 10) / 8;
                p.Y = (p.Y - 10) / 8;
                for (int i = Math.Max(0, p.X - 20); i < Math.Min(28, p.X + 20); i++) {
                    for (int j = Math.Max(0, p.Y - 20); j < Math.Min(28, p.Y + 20); j++) {
                        double dx = p.X - i;
                        double dy = p.Y - j;
                        double dist = Math.Sqrt(dx * dx + dy * dy);
                        double newIntensity = 255 - Math.Min(255, (Math.Tanh(2.5 * dist - 3.0) + 1.0) / 2.0 * 255);
                        double v = intensity[i, j] * 255;
                        if (newIntensity > v) {
                            int value = 255 - (int)(newIntensity);
                            intensity[i, j] = newIntensity / 255;
                            g.FillRectangle(new SolidBrush(Color.FromArgb(255, value, value, value)), i * 8 + 10, j * 8 + 10, 8, 8);
                        }
                    }
                }
            }
        }

        private void Form1_MouseUp (object sender, MouseEventArgs e) {
            isMouseDown = false;
        }

        private void clearBtn_Click (object sender, EventArgs e) {
            g = this.CreateGraphics();
            g.FillRectangle(Brushes.White, 10, 10, 28 * 8, 28 * 8);
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    intensity[i, j] = 0;
        }

        private void classifyBtn_Click (object sender, EventArgs e) {
            g = this.CreateGraphics();
            double[,,] curr = new double[1, 28, 28];
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    curr[0, j, i] = intensity[i, j];
            
            double[] prediction = network.getProbabilities(curr);

            for (int i = 0; i < 10; i++) {
                g.FillRectangle(Brushes.LightGray, 10 + i * 23, 295, 21, 60);
            }
            for (int i = 0; i < 10; i++) {
                g.FillRectangle(Brushes.Black, 10 + i * 23, 295, 21, (int)(60 * prediction[i]));
            }

            // first pooled maps
            int size = (28 - 2 - 2) / 2;
            for (int i = 0; i < 16; i++) {
                Bitmap bmp = new Bitmap(size, size);
                Neuron[] neurons = network.layers[3].neurons;
                double max = 0;
                for (int j = 0; j < size * size; j++)
                    max = Math.Max(max, neurons[i * size * size + j].activated);
                for (int m = 0; m < size; m++) {
                    for (int n = 0; n < size; n++) {
                        int val = (int)(255 * (neurons[i * size * size + m * size + n].activated / max));
                        bmp.SetPixel(n, m, Color.FromArgb(255, val, val, val));
                    }
                }
                g.DrawImage(new Bitmap(bmp, size * 2, size * 2), 250 + (i % 8) * (size * 2 + 2), 160 + (i / 8) * (size * 2 + 2));
            }

            size = ((28 - 2 - 2) / 2 - 2 - 2) / 2;
            for (int i = 0; i < 32; i++) {
                Bitmap bmp = new Bitmap(size, size);
                Neuron[] neurons = network.layers[6].neurons;
                double max = 0;
                for (int j = 0; j < size * size; j++)
                    max = Math.Max(max, neurons[i * size * size + j].activated);
                for (int m = 0; m < size; m++) {
                    for (int n = 0; n < size; n++) {
                        int val = (int)(255 * (neurons[i * size * size + m * size + n].activated / max));
                        bmp.SetPixel(n, m, Color.FromArgb(255, val, val, val));
                    }
                }
                g.DrawImage(new Bitmap(bmp, size * 4, size * 4), 285 + (i % 8) * (size * 4 + 2), 220 + (i / 8) * (size * 4 + 2));
            }
            
            this.Text = "Classified as " + getMax(prediction);
        }

        private int getMax (double[] pred) {
            int ret = 0;
            for (int i = 1; i < pred.GetLength(0); i++)
                if (pred[i] > pred[ret])
                    ret = i;
            return ret;
        }
    }
}
