using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;

namespace HandWrite
{
    public class DataFactory
    {
        private const string TestDataFile = "Data/testimages.idx";
        private const string TrainingDataFile = "Data/trainimages.idx";
        private const string TestLabelsFile = "Data/testlabels.idx";
        private const string TrainingLabelsFile = "Data/trainlabels.idx";

        public List<Tuple<DenseMatrix, int>> TestData
        {
            get
            {
                return null;
            }
        }

        public IEnumerable<byte> GetNumbers(int[] numbers)
        {
            var trainingDataFilePath = GetFilePath(TrainingDataFile);
            int position = 0;
            using (FileStream fs = new FileStream(@"file.txt", FileMode.Open, FileAccess.Read))
            {
                fs.Seek(position, SeekOrigin.Begin);

                var b = new byte[1];
                fs.Read(b, 0, 1);

                yield return b[0];
            }
        }

        public IEnumerable<IList<Tuple<DenseMatrix, int>>> TrainingData(int batchSize)
        {
            var result = new List<Tuple<DenseMatrix, int>>();
            var trainingDataFilePath = GetFilePath(TrainingDataFile);
            var bytes = File.ReadAllBytes(trainingDataFilePath);

            int rowcount;
            int columnCount;
            var imageCount = GetMetadata(bytes, out rowcount, out columnCount);
            bytes = bytes.Skip(16).ToArray();//16 bits header
            var trainingLabelFilePath = GetFilePath(TrainingLabelsFile);
            var labelBytes = File.ReadAllBytes(trainingLabelFilePath).Skip(8).ToArray();//8 bits header

            var imgList = Enumerable.Range(0, imageCount);
            //shuffle
            imgList = imgList.OrderBy(a => Guid.NewGuid());
            var batchIndexor = 0;
            foreach (var index in imgList)
            {
                batchIndexor++;
                var labelByte = labelBytes.Skip(index).Take(1).ToList()[0];
                var lb = (int)labelByte;
                var imageBytes = bytes.Skip(rowcount * columnCount * index).Take(rowcount * columnCount).ToList();
                var oneImage = new DenseMatrix(rowcount, columnCount);

                for (int j = 0; j < imageBytes.Count; j++)
                {
                    var rowIndex = j / rowcount;
                    var colIndex = j % columnCount;
                    oneImage[rowIndex, colIndex] = Math.Round((double)imageBytes[j] / 255, 4);
                }
                result.Add(new Tuple<DenseMatrix, int>(oneImage, lb));
                if (batchIndexor == batchSize)
                {
                    batchIndexor = 0;
                    yield return result;
                    result.Clear();
                }
            }

        }



        public IEnumerable<Tuple<DenseMatrix, int>> TrainingData()
        {
            var trainingDataFilePath = GetFilePath(TrainingDataFile);
            var bytes = File.ReadAllBytes(trainingDataFilePath);

            int rowcount;
            int columnCount;
            var imageCount = GetMetadata(bytes, out rowcount, out columnCount);
            bytes = bytes.Skip(16).ToArray();//16 bits header
            var trainingLabelFilePath = GetFilePath(TrainingLabelsFile);
            var labelBytes = File.ReadAllBytes(trainingLabelFilePath).Skip(8).ToArray();//8 bits header

            var imgList = Enumerable.Range(0, imageCount);
            //shuffle
            imgList = imgList.OrderBy(a => Guid.NewGuid());

            foreach (var index in imgList)
            {
                var labelByte = labelBytes.Skip(index).Take(1).ToList()[0];
                var lb = (int)labelByte;
                var imageBytes = bytes.Skip(rowcount * columnCount * index).Take(rowcount * columnCount).ToList();
                var oneImage = new DenseMatrix(rowcount, columnCount);

                for (int j = 0; j < imageBytes.Count; j++)
                {
                    var rowIndex = j / rowcount;
                    var colIndex = j % columnCount;
                    oneImage[rowIndex, colIndex] = Math.Round((double)imageBytes[j] / 255, 4);
                }
                yield return new Tuple<DenseMatrix, int>(oneImage, lb);
            }

        }

        private static int GetMetadata(byte[] bytes, out int rowcount, out int columnCount)
        {
            var icBytes = new byte[] { bytes[4], bytes[5], bytes[6], bytes[7] };
            if (BitConverter.IsLittleEndian)
                Array.Reverse(icBytes);
            var ic = BitConverter.ToInt32(icBytes, 0);


            var rcBytes = new byte[] { bytes[8], bytes[9], bytes[10], bytes[11] };
            if (BitConverter.IsLittleEndian)
                Array.Reverse(rcBytes);
            rowcount = BitConverter.ToInt32(rcBytes, 0);

            var ccBytes = new byte[] { bytes[12], bytes[13], bytes[14], bytes[15] };
            if (BitConverter.IsLittleEndian)
                Array.Reverse(ccBytes);
            columnCount = BitConverter.ToInt32(ccBytes, 0);
            return ic;
        }

        private string GetFilePath(string relPath)
        {
            return Path.Combine(AssemblyDirectory, relPath);
        }
        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }
    }
}
