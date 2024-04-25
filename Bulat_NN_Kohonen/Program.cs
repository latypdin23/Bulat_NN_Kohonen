using Bulat_NN_Kohonen.Classes;
using System;
using System.Collections.Generic;
using System.IO;


namespace Bulat_NN_Kohonen
{
    internal class Program
    {
        #region Информация о базе данных
        public static int trainCount = 60000;
        public static int testCount = 10000;
        public static int height = 28;
        public static int width = 28;
        #endregion
        public static List<Database> databaseForRecognition = new List<Database>();
        public static List<KohonenNeuralNetwork> kohonenNeuralNetwork = new List<KohonenNeuralNetwork>();
        public static int countOfNumbers = 10;

        public static void SeparateDatabase(ref int testValue, ref double[,] arrayForNumber, ref int[,] testArray, int index)
        {
            testValue++;
            for (int j = 0; j < height * width; j++)
            {
                arrayForNumber[testValue, j] = (double)testArray[index, j];
            };
        }
        public static void DownloadMNISTDatabase(string pixelFile, string labelFile, string pixelFile2, string labelFile2)
        {
            try
            {
                MNIST mnist = new MNIST(trainCount, testCount, height * width);
                mnist.trainImages = MNIST.LoadData(trainCount, pixelFile, labelFile, mnist.arr);
                Console.WriteLine("The training sample was loaded successfully\n");

                mnist.testImages = MNIST.LoadData(testCount, pixelFile2, labelFile2, mnist.arrT);
                Console.WriteLine("The test sample was loaded successfully\n");

                //dtb.arr = new double[dtb.trainCount, dtb.height * dtb.width];
                //dtb.arrT = new double[dtb.testCount, dtb.height * dtb.width];
                //dtb.label = new byte[dtb.trainCount];
                //dtb.labelT = new byte[dtb.testCount];
                //for (int i = 0; i < dtb.trainCount; i++)
                //{
                //    for (int j = 0; j < dtb.height * dtb.width; j++)
                //        dtb.arr[i, j] = (double)(mnist.arr[i, j]);
                //    dtb.label[i] = mnist.trainImages[i].label;
                //}
                //for (int i = 0; i < dtb.testCount; i++)
                //{
                //    for (int j = 0; j < dtb.height * dtb.width; j++)
                //        dtb.arrT[i, j] = (double)(mnist.arrT[i, j]);
                //    dtb.labelT[i] = mnist.testImages[i].label;
                //}


                // Загрузка отдельных изображений в разные массивы
                var learningSample = new int[countOfNumbers];
                var testSample = new int[countOfNumbers];

                for (int i = 0; i < countOfNumbers; i++)
                {
                    learningSample[i] = -1;
                    testSample[i] = -1;
                }

                // Загрузка отдельных изображений в разные массивы
                int[] k = new int[countOfNumbers]; int[] kT = new int[countOfNumbers];
                for (int i = 0; i < countOfNumbers; i++)
                {
                    k[i] = -1;
                    kT[i] = -1;

                    Database obj = new Database(7000, 2000, height, width);

                    databaseForRecognition.Add(obj);

                }

                for (int i = 0; i < trainCount; i++)
                {
                    SeparateDatabase(ref learningSample[mnist.trainImages[i].label], ref databaseForRecognition[mnist.trainImages[i].label].arrObj, ref mnist.arr, i);
                }
                for (int i = 0; i < testCount; i++)
                {
                    SeparateDatabase(ref testSample[mnist.testImages[i].label], ref databaseForRecognition[mnist.testImages[i].label].arrObjT, ref mnist.arrT, i);
                }

                for (int i = 0; i < countOfNumbers; i++)
                {
                    databaseForRecognition[i].countOfImagesForeachNumberTrain = learningSample[i];
                    databaseForRecognition[i].countOfImagesForeachNumberTest = testSample[i];

                    learningSample[i]++;
                    testSample[i]++;

                    //infoAboutDatabase.Rows.Add(i.ToString(), learningSample[i].ToString(), testSample[i].ToString());
                }
            }
            catch (Exception e)
            {
                return;
            }
        }
        public static void CreateKNN(int maxClusters)
        {
            for (int j = 0; j < countOfNumbers; j++)
            {
                KohonenNeuralNetwork knn = new KohonenNeuralNetwork(height * width, databaseForRecognition[j].countOfImagesForeachNumberTrain); ///rework
                knn.maxClusters = maxClusters;
                knn.w = new double[knn.maxClusters, knn.sizeOfVector];
                knn.d = new double[knn.maxClusters];
                
                kohonenNeuralNetwork.Add(knn);
            }
        }
        private static void CreateClusters(bool createAgain)
        {
            //CreateKNN(20);
            try
            {

                Random[] r = new Random[countOfNumbers];
                Random[] rand = new Random[countOfNumbers];
                for (int k = 0; k < countOfNumbers; k++)
                {
                    r[k] = new Random();
                    rand[k] = new Random();

                    for (int i = 0; i < kohonenNeuralNetwork[k].maxClusters; i++)
                    {
                        for (int j = 0; j < kohonenNeuralNetwork[k].sizeOfVector; j++)
                        {
                            kohonenNeuralNetwork[k].w[i, j] = r[k].NextDouble();
                        }
                    }
                }
                for (int k = 6; k < countOfNumbers; k++)
                {
                    double[,] pattern = new double[databaseForRecognition[k].countOfImagesForeachNumberTrain, kohonenNeuralNetwork[k].sizeOfVector];
                    for (int i = 0; i < databaseForRecognition[k].countOfImagesForeachNumberTrain; i++)
                    {
                        for (int j = 0; j < kohonenNeuralNetwork[k].sizeOfVector; j++)
                        {
                            pattern[i, j] = databaseForRecognition[k].arrObj[i, j];
                        }
                    }

                    kohonenNeuralNetwork[k].Training(pattern);

                    string fileName = "BigCluster" + k.ToString() + ".txt";
                    WritingClusterIntoFile(fileName, kohonenNeuralNetwork[k].w, kohonenNeuralNetwork[k].maxClusters);
                }
                Console.WriteLine("Clusters are built" + "\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                Console.WriteLine(ex.ToString());
            }
        }
        private static void ReadClustersFromFile(string fileName, int k)
        {
            try
            {
                string[] s = File.ReadAllLines(fileName);

                for (int i = 0; i < kohonenNeuralNetwork[k].maxClusters; i++)
                {
                    for (int j = 0; j < kohonenNeuralNetwork[k].sizeOfVector; j++)
                    {
                        kohonenNeuralNetwork[k].w[i, j] = Double.Parse(s[i * kohonenNeuralNetwork[k].sizeOfVector + j]);

                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }
        private static void EvclidDistanceForAllClusters(int vectorNumber, double[,] t, int maxClusters, double[,] w, double[] d)
        {
            for (int i = 0; i < maxClusters; i++)
            {
                d[i] = 0.0;
            }

            for (int i = 0; i < maxClusters; i++)
            {
                for (int j = 0; j < height * width; j++)
                {
                    d[i] += ((w[i, j] - t[vectorNumber, j])*(w[i, j] - t[vectorNumber, j]));
                }
            }
        }
        private static int MinDistance(double[] d, int maxClusters)
        {
            double min = d[0];
            int cl = 0;
            for (int i = 1; i < maxClusters; i++)
            {
                if (d[i] < min)
                {
                    min = d[i]; cl = i;
                }
            }
            return cl;
        }
        private static void VerifyClusters(string fileName)
        {
            try
            {
                using (FileStream aFile = new FileStream(fileName, FileMode.OpenOrCreate))
                {
                    using (StreamWriter swr = new StreamWriter(aFile))
                    {
                        aFile.Seek(0, SeekOrigin.End);

                        int[] countOfTrue = new int[countOfNumbers];
                        int[,] fMera = new int[countOfNumbers, countOfNumbers];
                        for (int k = 0; k < countOfNumbers; k++)
                        {
                            countOfTrue[k] = 0;
                            for (int h = 0; h < countOfNumbers; h++)
                            {
                                fMera[k, h] = 0;
                            }
                            double[,] test = new double[databaseForRecognition[k].countOfImagesForeachNumberTest, 
                                                        kohonenNeuralNetwork[k].sizeOfVector];
                            for (int i = 0; i < databaseForRecognition[k].countOfImagesForeachNumberTest; i++)
                            {
                                for (int j = 0; j < kohonenNeuralNetwork[k].sizeOfVector; j++)
                                {
                                    test[i, j] = databaseForRecognition[k].arrObjT[i, j];
                                }
                            }
                            for (int vecNum = 0; vecNum < databaseForRecognition[k].countOfImagesForeachNumberTest; vecNum++)
                            {
                                double[] minDist = new double[countOfNumbers];

                                for (int j = 0; j < countOfNumbers; j++)
                                {
                                    EvclidDistanceForAllClusters(vecNum, test, kohonenNeuralNetwork[j].maxClusters, kohonenNeuralNetwork[j].w, kohonenNeuralNetwork[j].d);
                                    int cluster = MinDistance(kohonenNeuralNetwork[j].d, kohonenNeuralNetwork[j].maxClusters);
                                    minDist[j] = kohonenNeuralNetwork[j].d[cluster];
                                }

                                double min = Double.MaxValue;
                                int res = -1;
                                for (int j = 0; j < countOfNumbers; j++)
                                {
                                    if (minDist[j] < min)
                                    {
                                        min = minDist[j];
                                        res = j;
                                    }
                                }
                                if (res == k) countOfTrue[k]++;
                                fMera[k, res]++;
                            }
                            float procOfTrueClusters = ((float)countOfTrue[k] / databaseForRecognition[k].countOfImagesForeachNumberTest) * 100;
                            Console.WriteLine(k.ToString(), procOfTrueClusters.ToString() + " %");
                            //MessageBox.Show(countOfTrue[k].ToString());
                        }
                        for (int i = 0; i < countOfNumbers; i++)
                        {
                            for (int j = 0; j < countOfNumbers; j++)
                            {

                                swr.WriteLine(fMera[i, j]);
                            }
                        }
                        swr.Close();
                    }
                    Console.WriteLine("Clusters are checked");
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }
        public static void WritingClusterIntoFile(string fileName, double[,] tempW, int maxClusterElement)
        {
            FileStream aFile = new FileStream(fileName, FileMode.OpenOrCreate);
            StreamWriter swr = new StreamWriter(aFile);
            aFile.Seek(0, SeekOrigin.End);
            for (int i = 0; i < maxClusterElement; i++)
            {
                for (int j = 0; j < height * width; j++)
                {
                    swr.WriteLine(Math.Round(tempW[i, j], 5));
                }

            }
            swr.Close();
        }
        static void Main(string[] args)
        {
            string pixelFileTrain = @"train-images.idx3-ubyte";
            string labelFileTrain = @"train-labels.idx1-ubyte";
            string pixelFileTest = @"t10k-images.idx3-ubyte";
            string labelFileTest = @"t10k-labels.idx1-ubyte";

            DownloadMNISTDatabase(pixelFileTrain,labelFileTrain,pixelFileTest,labelFileTest);
            CreateKNN(20);
            //CreateClusters(true);
            for (int k = 0; k < countOfNumbers; k++)
            {
                string fileName = "BigCluster" + k.ToString() + ".txt";
                ReadClustersFromFile(fileName,k);
            }
            VerifyClusters("VerifyKNN.txt");
            int a = 0;
        }
    }
}
