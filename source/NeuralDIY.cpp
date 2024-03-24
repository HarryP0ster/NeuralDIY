#include <map>
#include <array>
#include <vector>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <filesystem>
#include <unordered_map>

std::filesystem::path root;

class Matrix
{
public:
    Matrix(size_t Rows, size_t Cols, double Default = 0.0)
        : R(Rows), C(Cols)
    {
        M.resize(Rows * Cols);

        if (Default != 0.0)
            Fill(Default);
    }

    Matrix T() const
    {
        Matrix Out(C, R);

        for (size_t i = 0; i < R; i++)
        {
            for (size_t j = 0; j < C; j++)
            {
                Out[j][i] = (*this)[i][j];
            }
        }

        return Out;
    }

    void Fill(double Value)
    {
        std::fill(M.begin(), M.end(), Value);
    }

    size_t RowCount() const
    {
        return R;
    }

    size_t ColumnCount() const
    {
        return C;
    }

    double* operator[](size_t i)
    {
        return &M[i * C];
    }

    double const* operator[](size_t i) const
    {
        return &M[i * C];
    }

    Matrix operator+(const Matrix& M1) const
    {
        Matrix Out(R, C);

        for (size_t i = 0; i < R * C; i++)
            Out.M[i] = M[i] + M1.M[i];

        return Out;
    }

    Matrix operator-(const Matrix& M1) const
    {
        Matrix Out(R, C);

        for (size_t i = 0; i < R * C; i++)
            Out.M[i] = M[i] - M1.M[i];

        return Out;
    }

    Matrix operator*(const Matrix& M1) const
    {
        Matrix Out(RowCount(), M1.ColumnCount());

        assert(M1.RowCount() == ColumnCount());

        for (size_t i = 0; i < RowCount(); i++)
        {
            for (size_t j = 0; j < M1.RowCount(); j++)
            {
                for (size_t k = 0; k < M1.ColumnCount(); k++)
                {
                    Out[i][k] += (*this)[i][j] * M1[j][k];
                }
            }
        }

        return Out;
    }

public:
    std::vector<double> M;
    size_t C = 0;
    size_t R = 0;
};

class vector : public Matrix
{
public:
    vector(size_t Length, double Default = 0.0)
        : Matrix(1, Length, Default)
    {

    }

    vector(const std::vector<double>& Values)
        : Matrix(1, Values.size())
    {
        std::copy(Values.begin(), Values.end(), M.begin());
    }

    vector(Matrix& MM)
        : Matrix(1, MM.ColumnCount())
    {
        std::copy(MM.M.begin(), MM.M.end(), M.begin());
    }

    vector(Matrix&& MM)
        : Matrix(1, MM.ColumnCount())
    {
        M = std::move(MM.M);
    }

    size_t Length() const
    {
        return C;
    }

    double& operator[](size_t i)
    {
        return M[i];
    }

    double operator[](size_t i) const
    {
        return M[i];
    }
};

struct csv_data
{
    std::pair<char, const vector&> operator[](size_t i)
    {
        return std::pair<char, const vector&>(keys[i], values[i]);
    }

    void insert(std::pair<char, vector&&> pair)
    {
        keys.push_back(pair.first);
        values.push_back(pair.second);
    }

    size_t size() const
    {
        return keys.size();
    }

private:
    std::vector<char> keys;
    std::vector<vector> values;
};

Matrix Sigmoid(const Matrix& Input)
{
    Matrix Output(Input.RowCount(), Input.ColumnCount());

    for (size_t i = 0; i < Input.RowCount() * Input.ColumnCount(); i++)
        Output.M[i] = 1.0 / (std::exp(-Input.M[i]) + 1.0);

    return Output;
}

Matrix Relu(const Matrix& Input)
{
    Matrix Output(Input.RowCount(), Input.ColumnCount());

    for (size_t i = 0; i < Input.RowCount(); i++)
        Output.M[i] = std::max(Input.M[i], 0.0);

    return Output;
}

Matrix PropagateSigmoid(const Matrix& errors, const Matrix& output)
{
    assert(errors.RowCount() == output.RowCount() && errors.ColumnCount() == output.ColumnCount());

    Matrix temp(errors.RowCount(), errors.ColumnCount());

    for (size_t i = 0; i < errors.RowCount() * errors.ColumnCount(); i++)
        temp.M[i] = errors.M[i] * output.M[i] * (1.0 - output.M[i]); // e * g`(x)

    return temp;
}

Matrix PropagateRelu(const Matrix& errors, const Matrix& output)
{
    assert(errors.RowCount() == output.RowCount() && errors.ColumnCount() == output.ColumnCount());

    Matrix temp(errors.RowCount(), errors.ColumnCount());

    for (size_t i = 0; i < errors.RowCount() * errors.ColumnCount(); i++)
        temp.M[i] = errors.M[i] * (output.M[i] <= 0.0 ? 0.0 : 1.0); // e * g`(x)

    return temp;
}

void ReadCSV(const std::string from, csv_data& to)
{
    std::ifstream file;

    std::filesystem::path filepath = std::filesystem::path(root)
        .append("Datasets\\")
        .append(from);

    file.open(filepath);

    if (!file)
        return;

    std::string stream = "";
    while (file >> stream)
    {
        size_t i = 1;
        size_t j = 0;
        vector numbers(std::count(stream.cbegin(), stream.cend(), ',') + 1);
        while (i < stream.size())
        {
            std::string num = "";
            while (stream[i] != ',' && i < stream.size())
            {
                num += stream[i++];
            }

            numbers[j] = (atof(num.c_str()) / 255.0) * 0.99 + 0.01;
            i++;
            j++;
        }

        to.insert(std::pair<char, vector>(stream[0], std::move(numbers)));
        stream = "";
    }

    file.close();
}

struct NeuralNetwork
{
    Matrix WeightsIH;
    Matrix WeightsHO;

    double LearningRate = 0.0;

    NeuralNetwork(size_t Input, size_t Hidden, size_t Output, double LR)
        : LearningRate(LR), WeightsIH(Hidden, Input), WeightsHO(Output, Hidden)
    {
        for (size_t i = 0; i < Hidden * Input; i++)
        {
            WeightsIH.M[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) / 1000.0;
        }

        for (size_t i = 0; i < Hidden * Output; i++)
        {
            WeightsHO.M[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) / 1000.0;
        }
    }

    ~NeuralNetwork()
    {

    }

    vector Predict(const vector& InputV)
    {
        Matrix input_to_hidden = std::move(WeightsIH * InputV.T());
        Matrix hidden_output = std::move(Relu(input_to_hidden));
        Matrix hidden_to_output = std::move(WeightsHO * hidden_output);
        Matrix output = std::move(Sigmoid(hidden_to_output));

        return vector(std::move(output.T()));
    }

    void Train(const vector& InputV, const vector& ValidationV)
    {
        Matrix input_to_hidden = std::move(WeightsIH * InputV.T());
        Matrix hidden_output = std::move(Relu(input_to_hidden));
        Matrix hidden_to_output = std::move(WeightsHO * hidden_output);
        Matrix output = std::move(Sigmoid(hidden_to_output));
        Matrix errors(WeightsHO.RowCount(), 1);

        for (size_t j = 0; j < errors.RowCount(); j++)
            errors[j][0] = ValidationV[j] - output[j][0];

        Matrix hidden_errors = std::move(WeightsHO.T() * errors);
        Matrix OE = std::move(PropagateSigmoid(errors, output) * hidden_output.T());
        Matrix IE = std::move(PropagateRelu(hidden_errors, hidden_output) * InputV);

        for (size_t i = 0; i < WeightsHO.RowCount() * WeightsHO.ColumnCount(); i++)
            WeightsHO.M[i] += OE.M[i] * LearningRate;

        for (size_t i = 0; i < WeightsIH.RowCount() * WeightsIH.ColumnCount(); i++)
            WeightsIH.M[i] += IE.M[i] * LearningRate;
    }
};

std::pair<size_t, double> GetMaxIndex(const vector& V)
{
    size_t Index = 0u;
    double MaxVal = -1.0;
    
    for (size_t i = 0; i < V.Length(); i++)
    {
        if (V[i] > MaxVal)
        {
            MaxVal = V[i];
            Index = i;
        }
    }

    return { Index, MaxVal };
}

int main(int argc, const char** argv)
{
    srand(time(NULL));

    if (argc == 0)
        return -1;

    root = ((std::filesystem::path(argv[0])).parent_path())
        .parent_path()
        .append("");

    csv_data train;
    csv_data validation;

    ReadCSV("mnist_train.csv", train);
    ReadCSV("mnist_test.csv", validation);

    NeuralNetwork net(train[0].second.Length(), 200, 10, 0.01);
    for (size_t e = 0; e < 5; e++)
    {
        printf("epoch %d\n", int(e));

        for (size_t i = 0; i < train.size(); i++)
        {
            vector targets(10, 0.01);
            targets[train[i].first - '0'] = 0.99;

            net.Train(train[i].second, targets);
        }
    }

    size_t count = 0u;
    for (size_t i = 0; i < validation.size(); i++)
    {
        char label = validation[i].first;
        vector val_vector(validation[i].second);

        vector res = net.Predict(val_vector);

        std::pair<size_t, double> prediction = GetMaxIndex(res);

        printf("Expected %c : got %d, confidece %f\n", label, int(prediction.first), prediction.second);

        if (int(label - '0') == int(prediction.first))
        {
            count++;
        }
    }

    double accuracy = count / double(validation.size());
    printf("\nAccuracy %f", accuracy);

    return 0;
}