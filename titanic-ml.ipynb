{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "sunrise-colony",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:07.447367Z",
     "iopub.status.busy": "2021-04-30T11:28:07.446564Z",
     "iopub.status.idle": "2021-04-30T11:28:07.455525Z",
     "shell.execute_reply": "2021-04-30T11:28:07.454570Z"
    },
    "papermill": {
     "duration": 0.068143,
     "end_time": "2021-04-30T11:28:07.455752",
     "exception": false,
     "start_time": "2021-04-30T11:28:07.387609",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/titanic/train.csv\n",
      "/kaggle/input/titanic/test.csv\n",
      "/kaggle/input/titanic/gender_submission.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "composite-relaxation",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:07.555869Z",
     "iopub.status.busy": "2021-04-30T11:28:07.555014Z",
     "iopub.status.idle": "2021-04-30T11:28:08.689373Z",
     "shell.execute_reply": "2021-04-30T11:28:08.688443Z"
    },
    "papermill": {
     "duration": 1.189091,
     "end_time": "2021-04-30T11:28:08.689561",
     "exception": false,
     "start_time": "2021-04-30T11:28:07.500470",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "marine-protest",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:08.787303Z",
     "iopub.status.busy": "2021-04-30T11:28:08.786397Z",
     "iopub.status.idle": "2021-04-30T11:28:08.822204Z",
     "shell.execute_reply": "2021-04-30T11:28:08.821473Z"
    },
    "papermill": {
     "duration": 0.08764,
     "end_time": "2021-04-30T11:28:08.822390",
     "exception": false,
     "start_time": "2021-04-30T11:28:08.734750",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train = pd.read_csv('/kaggle/input/titanic/train.csv')\n",
    "test=pd.read_csv('/kaggle/input/titanic/test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "endangered-links",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:08.924400Z",
     "iopub.status.busy": "2021-04-30T11:28:08.923175Z",
     "iopub.status.idle": "2021-04-30T11:28:08.957249Z",
     "shell.execute_reply": "2021-04-30T11:28:08.957802Z"
    },
    "papermill": {
     "duration": 0.092193,
     "end_time": "2021-04-30T11:28:08.958034",
     "exception": false,
     "start_time": "2021-04-30T11:28:08.865841",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Moran, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330877</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>McCarthy, Mr. Timothy J</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17463</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Palsson, Master. Gosta Leonard</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>349909</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>347742</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>237736</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "5            6         0       3   \n",
       "6            7         0       1   \n",
       "7            8         0       3   \n",
       "8            9         1       3   \n",
       "9           10         1       2   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "5                                   Moran, Mr. James    male   NaN      0   \n",
       "6                            McCarthy, Mr. Timothy J    male  54.0      0   \n",
       "7                     Palsson, Master. Gosta Leonard    male   2.0      3   \n",
       "8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   \n",
       "9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  \n",
       "5      0            330877   8.4583   NaN        Q  \n",
       "6      0             17463  51.8625   E46        S  \n",
       "7      1            349909  21.0750   NaN        S  \n",
       "8      2            347742  11.1333   NaN        S  \n",
       "9      0            237736  30.0708   NaN        C  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "square-emphasis",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.054065Z",
     "iopub.status.busy": "2021-04-30T11:28:09.052784Z",
     "iopub.status.idle": "2021-04-30T11:28:09.075307Z",
     "shell.execute_reply": "2021-04-30T11:28:09.075923Z"
    },
    "papermill": {
     "duration": 0.073468,
     "end_time": "2021-04-30T11:28:09.076166",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.002698",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>897</td>\n",
       "      <td>3</td>\n",
       "      <td>Svensson, Mr. Johan Cervin</td>\n",
       "      <td>male</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7538</td>\n",
       "      <td>9.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>898</td>\n",
       "      <td>3</td>\n",
       "      <td>Connolly, Miss. Kate</td>\n",
       "      <td>female</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330972</td>\n",
       "      <td>7.6292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>899</td>\n",
       "      <td>2</td>\n",
       "      <td>Caldwell, Mr. Albert Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>248738</td>\n",
       "      <td>29.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>900</td>\n",
       "      <td>3</td>\n",
       "      <td>Abrahim, Mrs. Joseph (Sophie Halaut Easu)</td>\n",
       "      <td>female</td>\n",
       "      <td>18.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2657</td>\n",
       "      <td>7.2292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>901</td>\n",
       "      <td>3</td>\n",
       "      <td>Davies, Mr. John Samuel</td>\n",
       "      <td>male</td>\n",
       "      <td>21.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>A/4 48871</td>\n",
       "      <td>24.1500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "5          897       3                    Svensson, Mr. Johan Cervin    male   \n",
       "6          898       3                          Connolly, Miss. Kate  female   \n",
       "7          899       2                  Caldwell, Mr. Albert Francis    male   \n",
       "8          900       3     Abrahim, Mrs. Joseph (Sophie Halaut Easu)  female   \n",
       "9          901       3                       Davies, Mr. John Samuel    male   \n",
       "\n",
       "    Age  SibSp  Parch     Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0     330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0     363272   7.0000   NaN        S  \n",
       "2  62.0      0      0     240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0     315154   8.6625   NaN        S  \n",
       "4  22.0      1      1    3101298  12.2875   NaN        S  \n",
       "5  14.0      0      0       7538   9.2250   NaN        S  \n",
       "6  30.0      0      0     330972   7.6292   NaN        Q  \n",
       "7  26.0      1      1     248738  29.0000   NaN        S  \n",
       "8  18.0      0      0       2657   7.2292   NaN        C  \n",
       "9  21.0      2      0  A/4 48871  24.1500   NaN        S  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "driving-sculpture",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.177501Z",
     "iopub.status.busy": "2021-04-30T11:28:09.176391Z",
     "iopub.status.idle": "2021-04-30T11:28:09.203887Z",
     "shell.execute_reply": "2021-04-30T11:28:09.204637Z"
    },
    "papermill": {
     "duration": 0.081498,
     "end_time": "2021-04-30T11:28:09.204866",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.123368",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "blond-thickness",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.305803Z",
     "iopub.status.busy": "2021-04-30T11:28:09.304509Z",
     "iopub.status.idle": "2021-04-30T11:28:09.323687Z",
     "shell.execute_reply": "2021-04-30T11:28:09.323011Z"
    },
    "papermill": {
     "duration": 0.06926,
     "end_time": "2021-04-30T11:28:09.323872",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.254612",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 11 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  418 non-null    int64  \n",
      " 1   Pclass       418 non-null    int64  \n",
      " 2   Name         418 non-null    object \n",
      " 3   Sex          418 non-null    object \n",
      " 4   Age          332 non-null    float64\n",
      " 5   SibSp        418 non-null    int64  \n",
      " 6   Parch        418 non-null    int64  \n",
      " 7   Ticket       418 non-null    object \n",
      " 8   Fare         417 non-null    float64\n",
      " 9   Cabin        91 non-null     object \n",
      " 10  Embarked     418 non-null    object \n",
      "dtypes: float64(2), int64(4), object(5)\n",
      "memory usage: 36.0+ KB\n"
     ]
    }
   ],
   "source": [
    "test.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "figured-latvia",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.431644Z",
     "iopub.status.busy": "2021-04-30T11:28:09.430679Z",
     "iopub.status.idle": "2021-04-30T11:28:09.434512Z",
     "shell.execute_reply": "2021-04-30T11:28:09.435048Z"
    },
    "papermill": {
     "duration": 0.064216,
     "end_time": "2021-04-30T11:28:09.435277",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.371061",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "excessive-postage",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.537389Z",
     "iopub.status.busy": "2021-04-30T11:28:09.536518Z",
     "iopub.status.idle": "2021-04-30T11:28:09.547099Z",
     "shell.execute_reply": "2021-04-30T11:28:09.547788Z"
    },
    "papermill": {
     "duration": 0.064869,
     "end_time": "2021-04-30T11:28:09.548032",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.483163",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "precious-pulse",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.650664Z",
     "iopub.status.busy": "2021-04-30T11:28:09.649730Z",
     "iopub.status.idle": "2021-04-30T11:28:09.656426Z",
     "shell.execute_reply": "2021-04-30T11:28:09.657125Z"
    },
    "papermill": {
     "duration": 0.060448,
     "end_time": "2021-04-30T11:28:09.657366",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.596918",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def bar_plot(feature):\n",
    "  survived=train[train['Survived']==1][feature].value_counts()\n",
    "  dead=train[train['Survived']==0][feature].value_counts()\n",
    "  dataset=pd.DataFrame([survived,dead])\n",
    "  dataset.index=['Survived','Dead']\n",
    "  return dataset.plot.bar(figsize=(10,5))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "traditional-habitat",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:09.762123Z",
     "iopub.status.busy": "2021-04-30T11:28:09.761283Z",
     "iopub.status.idle": "2021-04-30T11:28:10.053547Z",
     "shell.execute_reply": "2021-04-30T11:28:10.052536Z"
    },
    "papermill": {
     "duration": 0.344516,
     "end_time": "2021-04-30T11:28:10.053777",
     "exception": false,
     "start_time": "2021-04-30T11:28:09.709261",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Sex')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "dedicated-extra",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:10.201564Z",
     "iopub.status.busy": "2021-04-30T11:28:10.166153Z",
     "iopub.status.idle": "2021-04-30T11:28:10.398825Z",
     "shell.execute_reply": "2021-04-30T11:28:10.398091Z"
    },
    "papermill": {
     "duration": 0.29419,
     "end_time": "2021-04-30T11:28:10.399017",
     "exception": false,
     "start_time": "2021-04-30T11:28:10.104827",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4UhHjx7Vrbfequeff17S6Quj283vDQCi884772jlypWqr6+X2+3WqVOnlJSUpHfeecfu0uBQ/MsER4qNPft3hBMnTohRcADR2rBhg37/+99r0qRJ+utf/6qCggLNmzfP7rLgYAQtONKMGTOUl5enkydPaseOHVqwYIFuvvlmu8sCMAhcfPHF6uzslMvl0i233KI333zT7pLgYAwdwpHuuusuvfTSSzpx4oT+8pe/6Pbbb9ecOXPsLguAw525Wz569GhVVVUpPT1dx48ft7kqOBnfOoQjff3110pPT7e7DACDTHl5uaZNm6YjR45o+fLlam5u1q9+9St+kUOvEbTgSNOmTdPEiRN10003aebMmfJ6vXaXBADAPyFowZGCwaDeeOMN7dy5U++9955mzJihm266SVdddZXdpQFwsNbWVpWWluqrr77Sxo0bdejQIR0+fFjTp0+3uzQ4FJPh4UgxMTG69tprtWnTJlVUVMjlciknJ8fusgA43Nq1axUMBvXhhx9KksaMGaPHHnvM5qrgZEyGh2M1NTWpvLxcO3fuVEtLi+699167SwLgcB999JGKior01ltvSZJGjBjBsw4RFYIWHOmee+7R+++/r+nTp2vVqlU8hwxAn/ifj9tpb29njT5EhaAFR/rxj3+skpISDRs2zO5SAAwifr9fpaWl6ujo0Lvvvqunn35aWVlZdpcFB2MyPBylo6NDHo9Hra2t52wfPnx4P1cEYDA5deqUnnjiCVVVVUmSrr32Wi1atOifnkYBRIp3Dhxl/vz52rlzp6666iq5XC5ZlnXWn//4xz/sLhGAQ/3tb3/TU089pU8++USSlJGRoR/+8IeELESFO1oAgCFv//79WrRokbKzs3XFFVfIsix98MEH2r59ux5//HFdccUVdpcIhyJowZE2b96sm266ST6fz+5SAAwCv/zlLzV37lzNmDHjrO179uzRjh07tGXLFpsqg9OxjhYcqaWlRfPmzdPPfvYzvfTSS2pvb7e7JAAO9umnn/5TyJKk6dOn69ChQzZUhMGCoAVHeuCBB/T666/rjjvu0J49e3TttdcqLy/P7rIAOFRX32Dm282IBjP84FgxMTHKysrS2LFj9dRTT+nFF19UQUGB3WUBcKBTp07p0KFD51wz69SpUzZUhMGCoAVHOrMq/I4dO3Ty5En99Kc/1Z49e+wuC4BDtbW16a677jpnm8vl6udqMJgwGR6OdM0112jGjBmaO3cuq8IDAAYsghYcJxgM6vnnn+ch0gCAAY/J8HCcmJgY/elPf7K7DAAAukXQgiNdffXVqqiosLsMAAC6xNAhHOmaa65RU1OThg0bpuHDh4cfwfPOO+/YXRoAAGEELTjS119/fc7t6enp/VwJAADnR9ACAAAwhHW04EjXXHPNOde2YegQADCQELTgSC+++GL47+3t7SorK1NsLG9nAMDAwtAhBo158+bphRdesLsMAADCWN4Bg8KXX36p+vp6u8sAAOAsjLXAkf77HK1QKKTOzk6tWrXK5qoAADgbQ4dwpDPLOxw/flwff/yxJk2apMmTJ9tcFQAAZyNowVFyc3O1cOFCXXbZZWpqatKcOXMUHx+vxsZG3XfffbrlllvsLhEAgDDmaMFRDh48qMsuu0yStHv3bk2cOFEvv/yyduzYoT/+8Y82VwcAwNkIWnAUr9cb/vv777+v6dOnS5LGjBlzznW1AACwE0ELjnP06FG1tbXpvffe0/e///3w9vb2dhurAgDgn/GtQzjKokWLNHfuXF1wwQWaOnWqJk2aJEk6cOCA0tLSbK4OAICzMRkejlNbW6u6ujpddtll4eHCo0ePKhgMErYAAAMKQQsAAMAQ5mgBAAAYQtACAAAwhKAFAABgCEELAADAEIIWAACAIf8PJBwSh5mqc+4AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Embarked')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "timely-flood",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:10.517871Z",
     "iopub.status.busy": "2021-04-30T11:28:10.511775Z",
     "iopub.status.idle": "2021-04-30T11:28:10.784776Z",
     "shell.execute_reply": "2021-04-30T11:28:10.783924Z"
    },
    "papermill": {
     "duration": 0.334665,
     "end_time": "2021-04-30T11:28:10.784997",
     "exception": false,
     "start_time": "2021-04-30T11:28:10.450332",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('SibSp')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "homeless-vancouver",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:10.932694Z",
     "iopub.status.busy": "2021-04-30T11:28:10.931536Z",
     "iopub.status.idle": "2021-04-30T11:28:11.123247Z",
     "shell.execute_reply": "2021-04-30T11:28:11.122410Z"
    },
    "papermill": {
     "duration": 0.281838,
     "end_time": "2021-04-30T11:28:11.123442",
     "exception": false,
     "start_time": "2021-04-30T11:28:10.841604",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Pclass')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "personal-evolution",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.272840Z",
     "iopub.status.busy": "2021-04-30T11:28:11.271737Z",
     "iopub.status.idle": "2021-04-30T11:28:11.277923Z",
     "shell.execute_reply": "2021-04-30T11:28:11.277190Z"
    },
    "papermill": {
     "duration": 0.097466,
     "end_time": "2021-04-30T11:28:11.278117",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.180651",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[     PassengerId  Survived  Pclass  \\\n",
       " 0              1         0       3   \n",
       " 1              2         1       1   \n",
       " 2              3         1       3   \n",
       " 3              4         1       1   \n",
       " 4              5         0       3   \n",
       " ..           ...       ...     ...   \n",
       " 886          887         0       2   \n",
       " 887          888         1       1   \n",
       " 888          889         0       3   \n",
       " 889          890         1       1   \n",
       " 890          891         0       3   \n",
       " \n",
       "                                                   Name     Sex   Age  SibSp  \\\n",
       " 0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       " 1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       " 2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       " 3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       " 4                             Allen, Mr. William Henry    male  35.0      0   \n",
       " ..                                                 ...     ...   ...    ...   \n",
       " 886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       " 887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       " 888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       " 889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       " 890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       " \n",
       "      Parch            Ticket     Fare Cabin Embarked  \n",
       " 0        0         A/5 21171   7.2500   NaN        S  \n",
       " 1        0          PC 17599  71.2833   C85        C  \n",
       " 2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       " 3        0            113803  53.1000  C123        S  \n",
       " 4        0            373450   8.0500   NaN        S  \n",
       " ..     ...               ...      ...   ...      ...  \n",
       " 886      0            211536  13.0000   NaN        S  \n",
       " 887      0            112053  30.0000   B42        S  \n",
       " 888      2        W./C. 6607  23.4500   NaN        S  \n",
       " 889      0            111369  30.0000  C148        C  \n",
       " 890      0            370376   7.7500   NaN        Q  \n",
       " \n",
       " [891 rows x 12 columns],\n",
       "      PassengerId  Pclass                                          Name  \\\n",
       " 0            892       3                              Kelly, Mr. James   \n",
       " 1            893       3              Wilkes, Mrs. James (Ellen Needs)   \n",
       " 2            894       2                     Myles, Mr. Thomas Francis   \n",
       " 3            895       3                              Wirz, Mr. Albert   \n",
       " 4            896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)   \n",
       " ..           ...     ...                                           ...   \n",
       " 413         1305       3                            Spector, Mr. Woolf   \n",
       " 414         1306       1                  Oliva y Ocana, Dona. Fermina   \n",
       " 415         1307       3                  Saether, Mr. Simon Sivertsen   \n",
       " 416         1308       3                           Ware, Mr. Frederick   \n",
       " 417         1309       3                      Peter, Master. Michael J   \n",
       " \n",
       "         Sex   Age  SibSp  Parch              Ticket      Fare Cabin Embarked  \n",
       " 0      male  34.5      0      0              330911    7.8292   NaN        Q  \n",
       " 1    female  47.0      1      0              363272    7.0000   NaN        S  \n",
       " 2      male  62.0      0      0              240276    9.6875   NaN        Q  \n",
       " 3      male  27.0      0      0              315154    8.6625   NaN        S  \n",
       " 4    female  22.0      1      1             3101298   12.2875   NaN        S  \n",
       " ..      ...   ...    ...    ...                 ...       ...   ...      ...  \n",
       " 413    male   NaN      0      0           A.5. 3236    8.0500   NaN        S  \n",
       " 414  female  39.0      0      0            PC 17758  108.9000  C105        C  \n",
       " 415    male  38.5      0      0  SOTON/O.Q. 3101262    7.2500   NaN        S  \n",
       " 416    male   NaN      0      0              359309    8.0500   NaN        S  \n",
       " 417    male   NaN      1      1                2668   22.3583   NaN        C  \n",
       " \n",
       " [418 rows x 11 columns]]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_test_data=[train,test]\n",
    "train_test_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "bulgarian-approval",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.408450Z",
     "iopub.status.busy": "2021-04-30T11:28:11.407771Z",
     "iopub.status.idle": "2021-04-30T11:28:11.412243Z",
     "shell.execute_reply": "2021-04-30T11:28:11.411598Z"
    },
    "papermill": {
     "duration": 0.077115,
     "end_time": "2021-04-30T11:28:11.412417",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.335302",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "for dataset in train_test_data:\n",
    "  dataset['Title']=dataset['Name'].str.extract(pat=' ([A-Za-z]+)\\.',expand=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "interesting-vanilla",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.529239Z",
     "iopub.status.busy": "2021-04-30T11:28:11.528175Z",
     "iopub.status.idle": "2021-04-30T11:28:11.533316Z",
     "shell.execute_reply": "2021-04-30T11:28:11.532601Z"
    },
    "papermill": {
     "duration": 0.066251,
     "end_time": "2021-04-30T11:28:11.533497",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.467246",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mr          517\n",
       "Miss        182\n",
       "Mrs         125\n",
       "Master       40\n",
       "Dr            7\n",
       "Rev           6\n",
       "Col           2\n",
       "Major         2\n",
       "Mlle          2\n",
       "Sir           1\n",
       "Lady          1\n",
       "Jonkheer      1\n",
       "Countess      1\n",
       "Ms            1\n",
       "Capt          1\n",
       "Mme           1\n",
       "Don           1\n",
       "Name: Title, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Title'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "serious-class",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.652644Z",
     "iopub.status.busy": "2021-04-30T11:28:11.651823Z",
     "iopub.status.idle": "2021-04-30T11:28:11.657257Z",
     "shell.execute_reply": "2021-04-30T11:28:11.656472Z"
    },
    "papermill": {
     "duration": 0.070052,
     "end_time": "2021-04-30T11:28:11.657423",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.587371",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mr        240\n",
       "Miss       78\n",
       "Mrs        72\n",
       "Master     21\n",
       "Col         2\n",
       "Rev         2\n",
       "Dr          1\n",
       "Ms          1\n",
       "Dona        1\n",
       "Name: Title, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test['Title'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "incorrect-document",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.778833Z",
     "iopub.status.busy": "2021-04-30T11:28:11.778099Z",
     "iopub.status.idle": "2021-04-30T11:28:11.782047Z",
     "shell.execute_reply": "2021-04-30T11:28:11.781380Z"
    },
    "papermill": {
     "duration": 0.068074,
     "end_time": "2021-04-30T11:28:11.782229",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.714155",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "title_mapping = {\"Mr\": 0, \"Miss\": 1, \"Mrs\": 2, \n",
    "                 \"Master\": 3, \"Dr\": 3, \"Rev\": 3, \"Col\": 3, \"Major\": 3, \"Mlle\": 3,\"Countess\": 3,\n",
    "                 \"Ms\": 3, \"Lady\": 3, \"Jonkheer\": 3, \"Don\": 3, \"Dona\" : 3, \"Mme\": 3,\"Capt\": 3,\"Sir\": 3 }\n",
    "for dataset in train_test_data:\n",
    "    dataset['Title'] = dataset['Title'].map(title_mapping)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "inappropriate-failing",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:11.897069Z",
     "iopub.status.busy": "2021-04-30T11:28:11.896302Z",
     "iopub.status.idle": "2021-04-30T11:28:12.109335Z",
     "shell.execute_reply": "2021-04-30T11:28:12.108640Z"
    },
    "papermill": {
     "duration": 0.272353,
     "end_time": "2021-04-30T11:28:12.109536",
     "exception": false,
     "start_time": "2021-04-30T11:28:11.837183",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAloAAAFYCAYAAACLe1J8AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Z1A+gAAAACXBIWXMAAAsTAAALEwEAmpwYAAAabElEQVR4nO3de5SV9X3v8c9cHC5yR8ARbY2SKCuuZRJp41qNZx0Ris3BS3UphtW4WqusNsmqTdWKN7AStSPQxBw1pLEmJ42t2opR8CxpWZyYmOWpl2JbSowJIabKyGUYroUBZvb5I3FOrAgDw889m3m9/hHm2fM8X3H29s3ze/az6yqVSiUAABxx9dUeAADgaCW0AAAKEVoAAIUILQCAQoQWAEAhQgsAoBChBQBQSGO1BziQ9vad6epymy8ObPToIWlr21HtMYCjjNcWeqq+vi4jRx673219OrS6uipCix7xcwKU4LWF3rJ0CABQiNACAChEaAEAFNKnr9ECAI5OnZ370t6+Mfv27an2KD3W2NiUkSPHpKGh5/kktACA9117+8YMHDg4xx57fOrq6qo9zkFVKpXs3Lkt7e0bc9xxzT3+PkuHAMD7bt++PTn22GE1EVlJUldXl2OPHXbIZ+CEFgBQFbUSWW87nHktHQIAVTd02KAMHHDks2R3x75s37arR4/92c9ez5133p6tW7dm+PDhufXWP8tJJ/1Kr44vtACAqhs4oDEXXPfkEd/vkoUXZXsPH7tgwd255JLLMm3aJ7Ns2f/O/Pl35ctfXtSr41s6BAD6vfb2zXnttVczZcq0JMmUKdPy2muvpr29vVf7dUYLoJ8rtWRzOA5lmQeOpPXr1+e448amoaEhSdLQ0JDjjhuTDRvWZ+TIkYe9377xzAKgakot2RyOQ1nmgVpg6RAA6PfGjRuXTZs2pLOzM0nS2dmZTZs2ZuzYcb3ar9ACAPq9kSNHZcKED2X58mVJkuXLl+WDHzytV8uGiaVDAIAkyQ033JwvfGFuvv71BzN06NDcdtuf9XqfQgsAqLrdHfuyZOFFRfbbU7/6qyfna1/7X0f0+EILAKi67dt2HZVvhHCNFgBAIUILAKAQoQUAUIjQAgAoRGgBABQitAAACnF7BwCg6kYOb0pj04Ajvt99ezrSvnXPQR93331fyrPPrkhr67p885uP5JRTJhyR4wstAKDqGpsG5Cd3XnrE93vKLY8nOXhonXPOf89ll12Rz372miN6fKEFAPR7Z575kSL7dY0WAEAhQgsAoJBDCq377rsvp512Wl577bUkySuvvJILL7ww06ZNy1VXXZW2trbuxx5oGwBAf9Dj0Pr3f//3vPLKKxk/fnySpKurKzfccEPmzJmTZcuWZdKkSVmwYMFBtwEA9Bc9uhh+z549ueOOO7Jw4cJceeWVSZJVq1ZlwIABmTRpUpLkiiuuyHnnnZe77777gNsAAP6rfXs6fvEOwSO/35740pfm59ln/082b27LH//xZzNs2PB861uP9fr4PQqte++9NxdeeGFOPPHE7q+1trbmhBNO6P79qFGj0tXVlS1bthxw24gRI3o83OjRQ3r8WPq3MWOGVnsE4AjpS8/nvjTL0WbDhvo0Nv7/hbXtO/clO/cVOdYvH+e9XH/9jbn++hsP+rj6+vpD+rk4aGitXLkyq1atyvXXX9/jnR4pbW070tVVed+PS20ZM2ZoNm7cXu0xoGb1tZjoK89nry1ldXV1Zd++rmqPcci6urre9XNRX1/3nieHDhpaL774YtasWZPzzjsvSfLWW2/l93//9/PpT38669at637c5s2bU19fnxEjRqS5ufk9twEA9BcHPZc2a9asPPfcc1mxYkVWrFiR448/Pn/1V3+Vq6++Ort3785LL72UJHnkkUdy/vnnJ0nOOOOM99wGANBfHPad4evr63PPPfdk7ty56ejoyPjx4zN//vyDbgMA6C8OObRWrFjR/euPfexjWbJkyX4fd6BtAAD9gTvDAwAU4kOlAYCqGzpiQAYe03TE97t7755s33Lwe2lt3bol8+bNyZtvvpFjjjkmJ574K7nhhpszcuTIXh1faAEAVTfwmKZc/ugfHvH9PjbjK9meg4dWXV1dZs68Mh/72M9vtn7//fdm0aL/mZtumtOr41s6BAD6vWHDhndHVpJ8+MNn5K233ur1foUWAMAv6erqyhNPPJ5PfOK/9XpfQgsA4Jd88YvzM3jwoFx66eW93pdrtAAAfuG++76UN974WVpavpj6+t6fjxJaAABJvvrV+/PDH/4g8+ffm6amI/MOSKEFAPR7P/nJmvz1X389J530K/mDP7gqSdLcfELuvntBr/YrtACAqtu9d08em/GVIvvtiVNOOTXPPffSET++0AIAqm77lo4e3e+q1njXIQBA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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Title')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "metric-backing",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:12.235759Z",
     "iopub.status.busy": "2021-04-30T11:28:12.234176Z",
     "iopub.status.idle": "2021-04-30T11:28:12.239477Z",
     "shell.execute_reply": "2021-04-30T11:28:12.238724Z"
    },
    "papermill": {
     "duration": 0.072416,
     "end_time": "2021-04-30T11:28:12.239661",
     "exception": false,
     "start_time": "2021-04-30T11:28:12.167245",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train.drop(['Name'],axis=1,inplace=True)\n",
    "test.drop(['Name'],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "intensive-validity",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:12.376420Z",
     "iopub.status.busy": "2021-04-30T11:28:12.375092Z",
     "iopub.status.idle": "2021-04-30T11:28:12.383917Z",
     "shell.execute_reply": "2021-04-30T11:28:12.383269Z"
    },
    "papermill": {
     "duration": 0.086531,
     "end_time": "2021-04-30T11:28:12.384113",
     "exception": false,
     "start_time": "2021-04-30T11:28:12.297582",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass     Sex   Age  SibSp  Parch   Ticket     Fare Cabin  \\\n",
       "0          892       3    male  34.5      0      0   330911   7.8292   NaN   \n",
       "1          893       3  female  47.0      1      0   363272   7.0000   NaN   \n",
       "2          894       2    male  62.0      0      0   240276   9.6875   NaN   \n",
       "3          895       3    male  27.0      0      0   315154   8.6625   NaN   \n",
       "4          896       3  female  22.0      1      1  3101298  12.2875   NaN   \n",
       "\n",
       "  Embarked  Title  \n",
       "0        Q      0  \n",
       "1        S      2  \n",
       "2        Q      0  \n",
       "3        S      0  \n",
       "4        S      2  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "beginning-apple",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:12.517688Z",
     "iopub.status.busy": "2021-04-30T11:28:12.516679Z",
     "iopub.status.idle": "2021-04-30T11:28:12.520379Z",
     "shell.execute_reply": "2021-04-30T11:28:12.521056Z"
    },
    "papermill": {
     "duration": 0.074063,
     "end_time": "2021-04-30T11:28:12.521299",
     "exception": false,
     "start_time": "2021-04-30T11:28:12.447236",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "sex_mapping={\"male\":0,\"female\":1}\n",
    "for dataset in train_test_data:\n",
    "  dataset['Sex']=dataset['Sex'].map(sex_mapping)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "eligible-review",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:12.646767Z",
     "iopub.status.busy": "2021-04-30T11:28:12.645960Z",
     "iopub.status.idle": "2021-04-30T11:28:12.660902Z",
     "shell.execute_reply": "2021-04-30T11:28:12.661860Z"
    },
    "papermill": {
     "duration": 0.080545,
     "end_time": "2021-04-30T11:28:12.662338",
     "exception": false,
     "start_time": "2021-04-30T11:28:12.581793",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Age'].fillna(train.groupby(['Title'])['Age'].transform(\"median\"),inplace=True)\n",
    "test['Age'].fillna(test.groupby(['Title'])['Age'].transform(\"median\"),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "palestinian-arrival",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:12.783311Z",
     "iopub.status.busy": "2021-04-30T11:28:12.782247Z",
     "iopub.status.idle": "2021-04-30T11:28:13.228549Z",
     "shell.execute_reply": "2021-04-30T11:28:13.229121Z"
    },
    "papermill": {
     "duration": 0.506502,
     "end_time": "2021-04-30T11:28:13.229328",
     "exception": false,
     "start_time": "2021-04-30T11:28:12.722826",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 20.0)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(x=train[train['Survived']==1]['Age'],shade='true')\n",
    "sns.kdeplot(x=train[train['Survived']==0]['Age'],shade='true')\n",
    "plt.xlim(0,20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "juvenile-element",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:13.360834Z",
     "iopub.status.busy": "2021-04-30T11:28:13.360104Z",
     "iopub.status.idle": "2021-04-30T11:28:13.628839Z",
     "shell.execute_reply": "2021-04-30T11:28:13.628233Z"
    },
    "papermill": {
     "duration": 0.342462,
     "end_time": "2021-04-30T11:28:13.629039",
     "exception": false,
     "start_time": "2021-04-30T11:28:13.286577",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20.0, 40.0)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(x=train[train['Survived']==1]['Age'],shade='true')\n",
    "sns.kdeplot(x=train[train['Survived']==0]['Age'],shade='true')\n",
    "plt.xlim(20,40)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "pharmaceutical-questionnaire",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:13.811201Z",
     "iopub.status.busy": "2021-04-30T11:28:13.796542Z",
     "iopub.status.idle": "2021-04-30T11:28:14.040989Z",
     "shell.execute_reply": "2021-04-30T11:28:14.039647Z"
    },
    "papermill": {
     "duration": 0.354154,
     "end_time": "2021-04-30T11:28:14.041186",
     "exception": false,
     "start_time": "2021-04-30T11:28:13.687032",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40.0, 60.0)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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sGABXnjhBeTk5GDlypUAgLlz52LevHkYM2YMysvL8dlnn2Hq1KkAgIceeghDhw4FALz66qvYvn07tFothBC4++67cf3116tVPhERnSdJCNG/25iyiE1q2cM6s2cg1AiwzmwbCHUOyCY1IiIiBg4REamCgUNERKpg4BARkSoYOEREpAoGDhERqYKBQ0REqmDgEBGRKnocOFu2bEEymezNWoiIaBDrceA8++yzuP7667Fs2TJ89tlnvVkTERENQj0OnDfffBMvvvgiDAYDHn74YZSUlOCFF17A4cOHe7M+IiIaJM7qHs5ll12G+fPn409/+hMWL16MTZs2YcqUKfjBD36AN998E6lUqrfqJCKiAe6sR4uuqanBm2++iTfffBOSJGHevHnw+XxYu3YtNm/ejOeee6436iQiogGux4Gzdu1arF+/HgcPHsQtt9yCVatW4corr8ysLykpwbXXXtsbNRIR0SDQ48D585//jHvvvReTJ0+GXq/vst5kMmH16tVZLY6IiAaPHt/DmThxIm655ZYuYbNmzZrM95wQjYiITqfHgfP888+fcvkvf/nLrBVDRESDV7dNah9++CEAQFEU/PWvf0XHCUIPHz4Mi8XSe9UREdGg0W3g/OxnPwMAxONxLFy4MLNckiTk5eXh8ccf773qiIho0Og2cLZu3QoAeOyxx7Bq1apeL4iIiAanHt/DYdgQEdH5OOMVzi233IK3334bAHDDDTdAkqRTbvf+++9nvTAiIhpczhg4y5cvz3z/5JNP9noxREQ0eJ0xcMaPH5/5fuLEib1eDBERDV49voezZs0a/P3vfwcAfPrpp7jxxhtx8803Y9euXb1WHBERDR49DpwXX3wRhYWFAICnnnoK99xzDx588EGsWLGi14ojIqLBo8eB09raCpvNhlAohK+++gqzZ8/GjBkzUF1d3Zv1ERHRINHjwTt9Ph927tyJ/fv3Y/z48ZBlGaFQCLIs92Z9REQ0SPQ4cB577DHMmzcPer0ezz77LADgvffew5gxY3qtOCIiGjx6HDg33HADtm3b1mlZaWkpSktLs14UERENPmc142drayuqq6vR1tbWafk111yT1aKIiGjw6XHgvP7661i2bBnMZjOMRmNmuSRJePfdd3ulOCIiGjx6HDhPP/00nnnmGdxwww29WQ8REQ1SPe4WrSgKZ/QkIqJz1uPAmTt3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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(x=train[train['Survived']==1]['Age'],shade='true')\n",
    "sns.kdeplot(x=train[train['Survived']==0]['Age'],shade='true')\n",
    "plt.xlim(40,60)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "killing-kenya",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:14.199618Z",
     "iopub.status.busy": "2021-04-30T11:28:14.194182Z",
     "iopub.status.idle": "2021-04-30T11:28:14.452518Z",
     "shell.execute_reply": "2021-04-30T11:28:14.453069Z"
    },
    "papermill": {
     "duration": 0.349415,
     "end_time": "2021-04-30T11:28:14.453306",
     "exception": false,
     "start_time": "2021-04-30T11:28:14.103891",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60.0, 80.0)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(x=train[train['Survived']==1]['Age'],shade='true')\n",
    "sns.kdeplot(x=train[train['Survived']==0]['Age'],shade='true')\n",
    "plt.xlim(60,80)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "physical-starter",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:14.582776Z",
     "iopub.status.busy": "2021-04-30T11:28:14.582014Z",
     "iopub.status.idle": "2021-04-30T11:28:14.600244Z",
     "shell.execute_reply": "2021-04-30T11:28:14.600757Z"
    },
    "papermill": {
     "duration": 0.085777,
     "end_time": "2021-04-30T11:28:14.600983",
     "exception": false,
     "start_time": "2021-04-30T11:28:14.515206",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "for dataset in train_test_data:\n",
    "  dataset.loc[dataset['Age']<=13,'Age']=1;\n",
    "  dataset.loc[(dataset['Age']>13) & (dataset['Age']<=18),'Age']=2;\n",
    "  dataset.loc[(dataset['Age']>18) & (dataset['Age']<=30),'Age']=3;\n",
    "  dataset.loc[(dataset['Age']>30) & (dataset['Age']<=50),'Age']=4;\n",
    "  dataset.loc[(dataset['Age']>50) & (dataset['Age']<=80),'Age']=5;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "married-wales",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:14.731840Z",
     "iopub.status.busy": "2021-04-30T11:28:14.731078Z",
     "iopub.status.idle": "2021-04-30T11:28:14.931023Z",
     "shell.execute_reply": "2021-04-30T11:28:14.930255Z"
    },
    "papermill": {
     "duration": 0.265989,
     "end_time": "2021-04-30T11:28:14.931201",
     "exception": false,
     "start_time": "2021-04-30T11:28:14.665212",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train['Age'].value_counts().plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "cleared-guess",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:15.073602Z",
     "iopub.status.busy": "2021-04-30T11:28:15.072841Z",
     "iopub.status.idle": "2021-04-30T11:28:15.306079Z",
     "shell.execute_reply": "2021-04-30T11:28:15.306710Z"
    },
    "papermill": {
     "duration": 0.3078,
     "end_time": "2021-04-30T11:28:15.306944",
     "exception": false,
     "start_time": "2021-04-30T11:28:14.999144",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Age')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "aboriginal-aggregate",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:15.479335Z",
     "iopub.status.busy": "2021-04-30T11:28:15.471093Z",
     "iopub.status.idle": "2021-04-30T11:28:15.705613Z",
     "shell.execute_reply": "2021-04-30T11:28:15.706268Z"
    },
    "papermill": {
     "duration": 0.327103,
     "end_time": "2021-04-30T11:28:15.706498",
     "exception": false,
     "start_time": "2021-04-30T11:28:15.379395",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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TZ06VUVFRb73jsjPz1evXr308ssvG7cZGmUfYMXFxVq7dq3vCvGKK65QamqqLrroIpuTwR9bt27VihUrfOfvyiuv1M9+9jMjrwyD1fbt2/XZZ59JOn3+brjhBpsT2YOyBwADMGcPAAag7AHAAJQ9ABiAjdAsMG3aNC1btqzFYwDaz7m2Sviun/zkJxYl6RgoewscPHjwrGP79u2zIQla4/rrr5fD4WhynDeM79jy8vIknX5byY8++si3Cmf79u0aOnQoZY/2s379eq1bt04HDhzQj3/8Y9/xyspK1tgHgbfeekuS9Oabb6q8vFwpKSnyer1688031aNHD5vToSXPPvusJGnq1KnauHGjEhISJEkFBQVauHChndFsQdkH0LBhw3TJJZdowYIFjd4ooVu3bhowYICNyeCP+Ph4SaffhOb3v/+97/jcuXN155136uc//7ld0dAKhYWFvqKXpISEBB06dMjGRPag7AMoPj5e8fHxys7O9h2rra3V8ePHFRISYmMytEZVVZVKS0t9O2CWlpaqqqrK5lTwV8+ePbV8+XLfDphvvfWWevbsaXMq61H2Fnj00Uc1f/58hYWF6bbbblNZWZnuv/9+3XfffXZHgx8mTZqk5ORk3XzzzZJOX+nff//99oaC3zIyMrRw4UKNHTtW0unnYjIyMmxOZT1eQWuB5ORkbdiwQe+88462bdump556ShMmTDByT+1gtWfPHu3cuVOSdN111zENFyQaGhq0fPlyptzElb0lzmyzunPnTt10003q0qWLnE5e4hBMBg4ceM697dGxhYSE6C9/+QtlL8reEn379tWUKVO0b98+Pf7446qurrY7Elrh448/VmZmpgoKCtTQ0OB7w3GWXgaHm2++WatXr1ZycnKjNyHv0qWLjamsxzSOBaqrq7V161YNGDBACQkJOnLkiL744gu2yA0SP/rRj/Tggw9q8ODBjf4iO7NaBx3bd/8iczgcvl/Wn3/+uY2prEfZAy24/fbb9Yc//MHuGECbMHFskzMrA9DxDR8+XB988IHdMdAOmnur0M6OOfsA2rt3b5NjZWVlFiZBW6xbt04vvviiunbtKpfLxZx9kNizZ49mzZolp9OpjIwMZWRkaMeOHYqKitLKlSuVmJhod0RLMY0TQAMHDlR8fLzO9S0+evSob+8OdGzffPPNOY8zZ9+x3X333br33ntVWVmpZcuW6dFHH9W4ceOUm5ur1157Ta+++qrdES3FlX0AxcfH67e//a3i4uLOGrvppptsSITzQakHpxMnTmjkyJGSpGXLlmncuHGSpBEjRuj555+3M5otmLMPoB/+8IdNXhXecsstFqdBa5WVlWn27NmaPHnyWdvlPvLIIzalgr+++xf1sGHDGo15PB6r49iOsg+gGTNmaMiQIeccmzNnjsVp0Frp6enq0aOHJk6cqHfffVcPP/yw7wVyBQUFNqdDS+Lj4317GD3zzDO+44cPHzZujb3EnD3QpHHjxuntt9+WdPoqcf78+Tp48KBWrFihlJQUbdiwwd6AOC8nT57UqVOnFBMTY3cUS3FlDzShrq7O97HD4VB6err69++vqVOnqqamxsZkaIuIiAjjil6i7IEmJSQk+DY/O2PGjBm6+uqrdeDAAXtCAeeJaRygCeXl5XI4HOd8V6q9e/eqX79+NqQCzg9lDwAGYBoHAAxA2QOAASh7ADAAZQ8ABvj/lZUQZUVgBsQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "P1=train[train['Pclass']==1]['Embarked'].value_counts()\n",
    "P2=train[train['Pclass']==2]['Embarked'].value_counts()\n",
    "P3=train[train['Pclass']==3]['Embarked'].value_counts()\n",
    "df=pd.DataFrame([P1,P2,P3])\n",
    "df.index=['1st Class','2nd Class','3rd Class']\n",
    "df.plot.bar(stacked=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "lyric-macintosh",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:15.846409Z",
     "iopub.status.busy": "2021-04-30T11:28:15.845731Z",
     "iopub.status.idle": "2021-04-30T11:28:15.851759Z",
     "shell.execute_reply": "2021-04-30T11:28:15.852328Z"
    },
    "papermill": {
     "duration": 0.078649,
     "end_time": "2021-04-30T11:28:15.852557",
     "exception": false,
     "start_time": "2021-04-30T11:28:15.773908",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Embarked'].fillna('S',inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "gentle-bacteria",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:15.998406Z",
     "iopub.status.busy": "2021-04-30T11:28:15.997331Z",
     "iopub.status.idle": "2021-04-30T11:28:16.031300Z",
     "shell.execute_reply": "2021-04-30T11:28:16.031848Z"
    },
    "papermill": {
     "duration": 0.106603,
     "end_time": "2021-04-30T11:28:16.032102",
     "exception": false,
     "start_time": "2021-04-30T11:28:15.925499",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  Sex  Age  SibSp  Parch            Ticket  \\\n",
       "0              1         0       3    0  3.0      1      0         A/5 21171   \n",
       "1              2         1       1    1  4.0      1      0          PC 17599   \n",
       "2              3         1       3    1  3.0      0      0  STON/O2. 3101282   \n",
       "3              4         1       1    1  4.0      1      0            113803   \n",
       "4              5         0       3    0  4.0      0      0            373450   \n",
       "..           ...       ...     ...  ...  ...    ...    ...               ...   \n",
       "886          887         0       2    0  3.0      0      0            211536   \n",
       "887          888         1       1    1  3.0      0      0            112053   \n",
       "888          889         0       3    1  3.0      1      2        W./C. 6607   \n",
       "889          890         1       1    0  3.0      0      0            111369   \n",
       "890          891         0       3    0  4.0      0      0            370376   \n",
       "\n",
       "        Fare Cabin Embarked  Title  \n",
       "0     7.2500   NaN        1      0  \n",
       "1    71.2833   C85        2      2  \n",
       "2     7.9250   NaN        1      1  \n",
       "3    53.1000  C123        1      2  \n",
       "4     8.0500   NaN        1      0  \n",
       "..       ...   ...      ...    ...  \n",
       "886  13.0000   NaN        1      3  \n",
       "887  30.0000   B42        1      1  \n",
       "888  23.4500   NaN        1      1  \n",
       "889  30.0000  C148        2      0  \n",
       "890   7.7500   NaN        3      0  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for dataset in train_test_data:\n",
    "  dataset.loc[dataset['Embarked']=='S','Embarked']=1;\n",
    "  dataset.loc[dataset['Embarked']=='C','Embarked']=2;\n",
    "  dataset.loc[dataset['Embarked']=='Q','Embarked']=3;\n",
    "train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "dated-howard",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:16.180392Z",
     "iopub.status.busy": "2021-04-30T11:28:16.179660Z",
     "iopub.status.idle": "2021-04-30T11:28:16.191347Z",
     "shell.execute_reply": "2021-04-30T11:28:16.190661Z"
    },
    "papermill": {
     "duration": 0.087656,
     "end_time": "2021-04-30T11:28:16.191517",
     "exception": false,
     "start_time": "2021-04-30T11:28:16.103861",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train['Fare'].fillna(train.groupby(['Pclass'])['Fare'].transform(\"mean\"),inplace=True)\n",
    "test['Fare'].fillna(test.groupby(['Pclass'])['Fare'].transform(\"mean\"),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "attached-rebound",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:16.344624Z",
     "iopub.status.busy": "2021-04-30T11:28:16.343738Z",
     "iopub.status.idle": "2021-04-30T11:28:16.896384Z",
     "shell.execute_reply": "2021-04-30T11:28:16.895640Z"
    },
    "papermill": {
     "duration": 0.633257,
     "end_time": "2021-04-30T11:28:16.896555",
     "exception": false,
     "start_time": "2021-04-30T11:28:16.263298",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7fa280e31590>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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WVkpy7KUlJSkK6+8UrNnzz5lCOihQ4c0a9YsBYNBSdLNN9+shx56SDfddJMuvPBC/fSnP9Xy5csHnf+Xv/ylvvCFL0iSVq9erYceekiS9JGPfERbtmyJ7BcMBnXZZZfprbfekiS98cYbuummm7RgwQItXbpU27Zti+x78OBBffrTn9YHPvAB3XLLLTp27NiIXjPB8SxN82Vo3+E2dfUE410KAAAAgBiaNm2aLMvS17/+df3tb3/T8ePHR3T8448/rgceeECvvfaaPvnJT6qmpka1tbWR7Rs3btSSJUtOOe7GG29UdXV1ZPm5555Tdna25syZo8bGRt1xxx364he/qJdeeklf//rXdffdd6ulpUWSdM8992jOnDnatm2b7rzzTv35z38eUc0Ex7PkdVuaUpimHTwkBwAAAJhQ0tLS9Nvf/laGYehb3/qWLr/8cn3hC1/Q0aPDywYf+9jHdN5558nlcik9PV3XX399JBDW1tbqvffe08KFC085bsmSJXr22WfV3d0tKRwwb7zxRknhMHr11VfrmmuukWmauuKKKzR37lz97W9/U11dnd5880195Stfkcfj0cUXX/y+5x8KwfEczJiUqZffbop3GQAAAABibPr06frud7+rv//979q4caOampr0ne98Z1jH+ny+QctLlizRE088IUmqrq7WokWLlJx86kM4p06dqunTp2vLli3q7u7Ws88+G+mZrKur0+bNm7VgwYLIf6+++qqOHDmipqYmZWRkKCUlJXKu4uLiEb1eHo5zDmZMytSW1w+rNxCS123FuxwAAAAAcTB9+nQtX75cjz76qM4//3z19PREtr1fL6RhGIOWP/ShD6mlpUW7d+9WdXW1vvGNb5z2WpWVlaqurpZt25oxY4amTp0qKRxGq6qq9OCDD55yzOHDh9XW1qaurq5IeKyrqzuljqHQ43gOkr0u+XJT9FZNS7xLAQAAABAj+/bt089//nM1NDRIkurr61VdXa0LL7xQ5eXlevnll1VXV6f29nb95Cc/OeP53G63Fi9erO9973s6fvy4rrjiitPu+9GPflTPP/+8fve736mysjKyfunSpdqyZYu2bt2qUCik3t5ebdu2TQ0NDZo0aZLmzp2rhx9+WH6/X6+88sqgh+wMB8HxHDFcFQAAAJhY0tLStH37dq1YsULz58/XJz7xCc2cOVOrV6/WFVdc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mTN5X0PUHTx4UF/+8pcVCoVk27amT5+ub37zmyooKKC9xcmYDo4AAAAAgPgbs0NVAQAAAABjA8ERAAAAADAkgiMAAAAAYEgERwAAAADAkAiOAAAAAIAhERwBAAAAAENyxbsAAACGY+HChTp69Kgsy4qs27x5swoLC+NYFQAAEwPBEQCQMH784x/rQx/60IiPcxxHjuPINBloAwDA2eA3KAAgIR0/flx33HGHLrvsMl188cW644471NDQENl+880366GHHtJNN92kCy+8UAcPHtS+fft0yy236JJLLlFFRYU2bdoUx1cAAEDiIDgCABKSbdtavny5tmzZoi1btsjr9er+++8ftM/jjz+uBx54QK+99ppycnL0uc99TpWVlXrhhRf00EMPae3atdq7d2+cXgEAAImDoaoAgIRx1113Re5xvOSSS/TII49Etn3xi1/UZz7zmUH7f+xjH9N5550nSdq6dasmTZqkj3/845Kk888/XxUVFdq8ebO+9KUvxegVAACQmAiOAICEsX79+sg9jt3d3VqzZo22bt2q48ePS5I6OzsVCoUi4dLn80WOPXz4sHbs2KEFCxZE1oVCIS1dujSGrwAAgMREcAQAJKSf//znqqmp0R/+8Afl5+dr9+7dWrZsmRzHiexjGEbke5/Pp4svvli/+MUv4lEuAAAJjXscAQAJqbOzU16vVxkZGWptbdWPfvSjIfe/9tprVVtbq8cee0yBQECBQEA7duzQvn37YlQxAACJi+AIAEhIn/3sZ9Xb26vLLrtMK1eu1FVXXTXk/mlpafrZz36mTZs26aqrrtKVV16p73//+/L7/TGqGACAxGU4A8f0AAAAAABwEnocAQAAAABDIjgCAAAAAIZEcAQAAAAADIngCAAAAAAYEsERAAAAADAkgiMAAAAAYEgERwAAAADAkAiOAAAAAIAhERwBAAAAAEP6/wGF492KhKl2PwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "virtual-diabetes",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:17.065151Z",
     "iopub.status.busy": "2021-04-30T11:28:17.052504Z",
     "iopub.status.idle": "2021-04-30T11:28:17.611711Z",
     "shell.execute_reply": "2021-04-30T11:28:17.610784Z"
    },
    "papermill": {
     "duration": 0.646653,
     "end_time": "2021-04-30T11:28:17.611924",
     "exception": false,
     "start_time": "2021-04-30T11:28:16.965271",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 20.0)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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s8/q8ccflyS9+eab+sIXvqBvf/vbOuuss47PJ/kQEklD+zv6ta81krklhiTNq/arPujVFefNU12lV04Ht8QAAAAAcOKZ1FDVBx54QE1NTXr00UdVXl6urVu3SpI2bNigjRs3asmSJVq3bp3eeOMNrVy5UpJ0991366STTpIkPfjggxoeHtamTZsy+/z617+u008/fao/z6T09sesW2IcCmvvoT61HhlQTYVH9VWlml9TqovOrFO5r4RJbAAAAABAks00zekZ//khfdChqkkjpYNdUe1r7UtXEyOKxQ3Nq7EmsWmoKlV9lU8uJ7fEmCyGVBQPjkVx4XgUF45H8eBYFBeOR3HheBSP5UtPnukuFLU5N+VndCihd1v79O6hsN452KcDnf0KlLkzs52e95FqVZa5qSYCAAAAwCTN6uCYuSXGIWvY6TuH+tQXjVv3Tazy6fxTq3XNxSfL45rVHxMAAAAAZtSsSlTxhDWJzd5D4cywU5fTrnnVpQpVleoTF56smoBX9hPklhgAAAAAMB1mTXD89v97U795r1s1Aa8aqnw6JVSuy5aEVOZzzXTXAAAAAGBOmzXB8fxTq3X5uSEmsQEAAACAaTZrbkzYUF1KaAQAAACAGTBrgiMAAAAAYGYQHAEAAAAABREcAQAAAAAFERwBAAAAAAURHAEAAAAABREcAQAAAAAFERwBAAAAoEht2rRJ27Ztm/L9PvLII7rnnnsm3d455T0AAAAAgDnutdde0ze+8Q3t3btXDodDixYt0l/+5V/qnHPOmdL3eeihh6Z0fx8UwREAAAAAjkE0GtVnP/tZPfDAA/rEJz6hRCKh1157TS6X65j2Y5qmTNOU3V78A0GLv4cAAAAAUERaWlokSWvWrJHD4ZDH49Fll12mM844Y8wQ0EOHDun0009XMpmUJN166616+OGHddNNN+ncc8/VP/3TP+mGG27I2/8///M/67Of/awkqampSQ8//LAk6ROf+ISef/75TLtkMqmLL75Yb731liTp17/+tW666SYtXbpU1113nV5++eVM24MHD+qWW27R+eefr9tuu029vb3H9JkJjgAAAABwDE455RQ5HA7de++9+vnPf66+vr5j2v6pp57S5s2b9frrr+sP/uAP1NLSov3792fW79ixQ2vXrh2z3bXXXqvm5ubM6xdeeEGVlZU666yz1NnZqTvvvFN33XWXXnnlFd17773auHGjenp6JEn33HOPzjrrLL388sv63Oc+p5/85CfH1GeCIwAAAAAcA7/frx/+8Iey2Wz6q7/6Ky1btkyf/exndeTIkUlt/8lPflKnnnqqnE6nysrKdPXVV2cC4f79+/Xee+/pqquuGrPd2rVr9dxzz2loaEiSFTCvvfZaSVYYvfzyy3XFFVfIbrfr0ksv1dlnn62f//znamtr029+8xv92Z/9mVwuly644IJx918IwREAAAAAjtHixYv1ta99Tb/4xS+0Y8cOdXV16a//+q8ntW0oFMp7vXbtWj399NOSpObmZq1YsUJer3fMdgsWLNDixYv1/PPPa2hoSM8991ymMtnW1qZdu3Zp6dKlma//+Z//0eHDh9XV1aXy8nL5fL7MvhoaGo7p8zI5DgAAAAB8CIsXL9YNN9ygJ554QmeeeaaGh4cz68arQtpstrzXl1xyiXp6evT222+rublZX/7yl4/6XmvWrFFzc7NSqZQ+8pGPaMGCBZKsMLpu3Tpt2bJlzDatra2KRCIaHBzMhMe2trYx/SiEiiMAAAAAHIN9+/bp+9//vjo6OiRJ7e3tam5u1rnnnqvGxka9+uqramtrU39/v7773e9OuL+SkhKtXr1aX//619XX16dLL730qG2vueYavfjii/rRj36kNWvWZJZfd911ev7557V7924ZhqFYLKaXX35ZHR0dmjdvns4++2w98sgjisfjeu211/Im2ZkMgiMAAAAAHAO/36833nhD69ev13nnnaff//3f12mnnaampiZdeumluuaaa3Tdddfphhtu0JVXXjmpfa5du1YvvfSSVq9eLafz6ANDa2trdd555+lXv/qVrrnmmszyUCikRx99VN/97ne1bNkyXXHFFfre976nVColSfrbv/1bvfHGG7rooou0bds2XX/99cf0mW2maZrHtMUMeeXNVsXixkx3A5ICAZ/C4cGZ7gbEsSg2HI/iwvEoHhyL4sLxKC4cj+KxfOnJM92FokbFEQAAAABQEMERAAAAAFAQwREAAAAAUBDBEQAAAABQEMERAAAAAFAQwREAAAAAUBDBEQAAAABQEMERAAAAAFCQc6Y7AAAAAABzwW2bn9GR8NCU77c64NVjf7VyUm1bWlrU1NSkcDisQCCgrVu3auHChR+6DwRHAACA2cw0JZnpR6Wfp2Qbsy79GEvJ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\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()\n",
    "plt.xlim(0,20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "iraqi-comparison",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:17.809640Z",
     "iopub.status.busy": "2021-04-30T11:28:17.807531Z",
     "iopub.status.idle": "2021-04-30T11:28:18.351553Z",
     "shell.execute_reply": "2021-04-30T11:28:18.350724Z"
    },
    "papermill": {
     "duration": 0.66872,
     "end_time": "2021-04-30T11:28:18.351731",
     "exception": false,
     "start_time": "2021-04-30T11:28:17.683011",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20.0, 40.0)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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JUFNKglSrQQAADhNdHZ26p/+6Z9077336uMf/7ji8bhefPFF+f3+ER3HcRw5jiNjHDyijuAIZBjJhD2tnVEdONKh9u642jpj6uiOqSDkU0VxQBXFQU1KTtRTURJURbH7PejnIwcAADDe7du3T5I0f/58SZJpmrrssssk9R8C+vbbb2v27Nl67bXX5PV6tXTpUr3//e/X73//e+3Zs0e33nqrduzYoSeeeCJ9/P/8z//U7373O33nO99RXV2dqqurtWrVKn384x/XnXfeqSuuuEKSlEgkdNlll+n73/++LrjgAr300kt68MEH9eabb2ry5Mm6++67dfHFF0uSDh48qLvuukuvvfaaZs6cqbPPPntE18xvscAwDTVhT+reyraumNq7YmrvjulwU7c6umNq63KDpt9nqrw4qEnFbpVyUmnIDZnJgFkQ9DJxDwAAQJ47++yzZZqmVq9erauvvlozZ85USUnJsPd/8skn9cgjj+jss89WT0+P/v3f/1379+/XWWedJUnatm2bbrzxxn77feITn1B9fX06OD733HMqKyvTBRdcoCNHjujmm2/Wxo0bNWvWLP3v//6vVq5cqZ/97GcqLy/X7bffrpkzZ+oHP/iBdu/erZtuukmzZ88e9jkTHIFRknVv5QBSFcv2ZJBs74rpLwdb1dHthszWzpjkOCorCqaHv1Ymg2WqallS6Of5lQAAAGOssLBQP/rRj/TII4/oy1/+so4fP67LL79c999//7D2/+QnP6l3v/vdkqSioiLNnj1b9fX1uvXWW7V//3799a9/1ZVXXtlvvwULFmjRokXq6elRKBTStm3b9IlPfEKSG0Yvv/xyffSjH5UkXXrppbrwwgv161//WhdffLFeeeUVbd68WX6/Xx/84AcHPH4uBEfgFMmsWNZWFAzYJhJLqL0rprauuNq7YzpwpEOv7WtWe7JiGY1bKi0MpKuUUyeXKOQz0uGyrCjAfZYAAACnwLnnnqsHH3xQkrR3717dcccd+td//ddhDQGtra3NWl6wYIEefPBB3Xrrraqvr9ecOXMUCvV/esDUqVN17rnn6tlnn9UVV1yhnTt3auvWrZKkhoYG7dixQ88++2y6fSKR0MUXX6yjR4+quLhY4XDvHB+TJ09WY2PjsK+X4AjkkaDfq6Dfq6qygbfHE7bau2PJcBlTw/EuHW/pdquYnTF19sRV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dy7XZ0OSfvrTn+qjH/1o+j96B5L52Xj99dd16NAhnX322SftnCeym266Sc8995x27typnTt3qqamRt///ve1bNkyMscIeRzHccbijd944w3Nnz9fZ511VvrRDlOmTNGmTZv0xz/+UWvXrs2aGjd179DChQv1H//xH6qurlZ3d7fq6ur0+uuvyzAM3XHHHZozZ85YXM64N1h/LF68WDfffHNWCExNCX3kyBHddNNNevLJJ9Xd3a3rr79e8XhcklRVVaW7775b55xzzphd03g1WF8sW7ZMa9askcfjUSKR0Pve9z596UtfUkFBQVZfSNKxY8d0xx13qKGhQYFAQOvWrdP73//+sbyscSvXv1WRSESXXnqpfvKTn/S7Lyvz36p//Md/VFNTkzwejwoLC3XnnXdq5syZY3A149vx48f1+c9/Xj09PTIMQyUlJVq9erUuuOAC7d27V3V1dWpvb1dxcbE2bNiQ/vdn+fLlWrlypaZPny7LsrR+/Xo9//zz6W2pCQ8wMoP1h9/vH/QzI/HZOFkG64/i4mLddtttsixLtm3r3HPP1T333JN+zAO/V42+XP9WSdLcuXN199136/LLL8/aL/PfqtWrV+u1116TYRjy+XxauXJlzpEtGL4rr7xS3/nOd3TeeeeROUZozIIjAAAAAGB8YBwhAAAAACAngiMAAAAAICeCIwAAAAAgJ4IjAAAAACAngiMAAAAAICeCIwAAAAAgp8GfWA0AQB658sordfz4cZmmmV63Y8cOHhgPAMApQHAEAIwb3/nOd3TJJZeMeD/HceQ4jgyDgTYAAJwIfoICAMaltrY23Xzzzfrwhz+sD37wg7r55pt1+PDh9PalS5fq61//uq677jq9973v1cGDB7V371599rOf1Yc+9CHNnTtX27dvH8MrAABg/CA4AgDGJdu29alPfUrPPvusnn32WQUCAa1fvz6rzZNPPqn77rtPf/zjH1VeXq4bb7xR8+fP129/+1t9/etf17p16/Tmm2+O0RUAADB+MFQVADBurFixIn2P44c+9CE9/PDD6W233HKLbrjhhqz2n/zkJ/Xud79bkrRr1y6dccYZ+vSnPy1Jes973qO5c+dqx44duvXWW0/RFQAAMD4RHAEA48amTZvS9zj29PRozZo12rVrl9ra2iRJXV1dsiwrHS5ra2vT+x46dEgvv/yyLrroovQ6y7J0zTXXnMIrAABgfCI4AgDGpR/84Afat2+ffvKTn6iyslKvv/66Fi1aJMdx0m08Hk/6dW1trT74wQ9q8+bNY3G6AACMa9zjCAAYl7q6uhQIBFRcXKzW1lZ9+9vfztn+7/7u77R//35t3bpV8Xhc8XhcL7/8svbu3XuKzhgAgPGL4AgAGJf+4R/+QdFoVB/+8Id17bXXatasWTnbFxYW6vvf/762b9+uWbNm6bLLLtNXv/pVxWKxU3TGAACMXx4nc0wPAAAAAAB9UHEEAAAAAOREcAQAAAAA5ERwBAAAAADkRHAEAAAAAOREcAQAAAAA5ERwBAAAAADkRHAEAAAAAOREcAQAAAAA5ERwBAAAAADk9P8DBA14pTGH6MoAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()\n",
    "plt.xlim(20,40)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "spiritual-supplier",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:18.531581Z",
     "iopub.status.busy": "2021-04-30T11:28:18.530776Z",
     "iopub.status.idle": "2021-04-30T11:28:19.078728Z",
     "shell.execute_reply": "2021-04-30T11:28:19.078005Z"
    },
    "papermill": {
     "duration": 0.652401,
     "end_time": "2021-04-30T11:28:19.079000",
     "exception": false,
     "start_time": "2021-04-30T11:28:18.426599",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40.0, 80.0)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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/XrNnDlz2A1ta+uU55lh74eRN3XqMfroowPlbgZEX1Qa+qOy0B+Vg76oLPRHZaE/KgcBPtkhp6pOmTJF9fX1am5uliQ1Nzervr5etbW1JeXmzZunDRs2yPM8tbe364UXXlBjY6Mk6Z133tGf//mf6x/+4R902mmnjcLXAAAAAACMliFNVb333nu1fPlyrV+/XpMmTdKaNWskSbfeequWLVumOXPmaMGCBXr77bd15ZVXSpKWLl2qE088UZK0atUq9fT0aMWKFVGdDz74oGbPnj3S3wcAAAAAMMIsY8y4mP/JVNXKwZSKykFfVBb6o7LQH5WDvqgs9EdloT8qB1NVkx1yqioAAAAA4OhGcAQAAAAAJCI4AgAAAAASERwBAAAAAIkIjgAAAACARARHAAAAAEAigiMAAAAAIBHBEQAAAACQiOAIAAAAAEhEcAQAAAAAJCI4AgAAAAASERwBAAAAAIkIjgAAAABQoVasWKF169aNeL1r167VXXfdNeTyqRFvAQAAAAAc4d544w1973vf0//93//JcRzNnDlTd999t84444wR/Zz77rtvROv7tAiOAAAAADAMnZ2duv3223XvvffqqquuUj6f1xtvvKFMJjOseowxMsbItit/ImjltxAAAAAAKkhLS4skqampSY7jqKqqShdeeKE+//nP95sC+sEHH2j27NkqFAqSpBtvvFEPPfSQrr/+ep155pn6p3/6Jy1cuLCk/n/+53/W7bffLklavny5HnroIUnSVVddpa1bt0blCoWCzjvvPP3P//yPJOlXv/qVrr/+es2dO1fXXnutXn311ajszp07dcMNN+gLX/iCbrrpJn388cfD+s4ERwAAAAAYht///d+X4zj6zne+o5///Ofav3//sPZ/9tlntXr1ar355pv64z/+Y7W0tGjHjh3R9o0bN2r+/Pn99rvmmmvU3NwcLb/88ss69thjddppp2nPnj267bbb9PWvf12vvfaavvOd72jZsmVqb2+XJN1111067bTT9Oqrr+ob3/iGfvKTnwyrzQRHAAAAABiGiRMn6l/+5V9kWZb++q//Wueff75uv/127du3b0j7f/nLX9bnPvc5pVIpHXPMMbr88sujQLhjxw799re/1WWXXdZvv/nz5+vFF19Ud3e3JD9gXnPNNZL8MHrxxRfrkksukW3buuCCC3T66afr5z//uXbt2qX//u//1p133qlMJqNzzjlnwPqTEBwBAAAAYJhOOeUU/c3f/I1+8YtfaOPGjdq7d6+++93vDmnfurq6kuX58+frueeekyQ1NzeroaFB1dXV/fY76aSTdMopp2jr1q3q7u7Wiy++GI1M7tq1S5s3b9bcuXOjP//1X/+ljz76SHv37tWkSZNUU1MT1TVjxoxhfV9ujgMAAAAAh+GUU07RwoUL9dRTT+nUU09VT09PtG2gUUjLskqWv/SlL6m9vV3vvfeempub9Zd/+ZeDflZTU5Oam5vleZ7+4A/+QCeddJIkP4wuWLBA999/f799PvzwQ33yySc6ePBgFB537drVrx1JGHEEAAAAgGHYvn27Hn/8ce3evVuS1NraqubmZp155pmqr6/X66+/rl27dunAgQN69NFHD1lfOp3WvHnz9OCDD2r//v264IILBi179dVX65VXXtG//uu/qqmpKVp/7bXXauvWrXrppZfkuq56e3v16quvavfu3Tr++ON1+umna+3atcrlcnrjjTdKbrIzFARHAAAAABiGiRMn6u2339aiRYt01lln6Y/+6I80a9YsLV++XBdccIGuvvpqXXvttVq4cKEuvfTSIdU5f/58/fKXv9S8efOUSg0+MXTatGk666yz9NZbb+nqq6+O1tfV1Wn9+vV69NFHdf755+uSSy7RY489Js/zJEl/93d/p7fffltf/OIXtW7dOl133XXD+s6WMcYMa48yaWvrlOeNi6Ye8aZOPUYffXSg3M2A6ItKQ39UFvqjctAXlYX+qCz0R+WYOvWYcjehojHiCAAAAABIRHAEAAAAACQiOAIAAAAAEhEcAQAAAACJCI4AAAAAgEQERwAAAABAIoIjAAAAACARwREAAAAAkChV7gYAAAAAwJHgptVbtK+je8Tr/ezkav3or68cUtmWlhYtX75cHR0dmjx5stasWaOTTz75sNtAcAQAAACAEbCvo1vf/foFI17v3d9/ZchlV65cqSVLlmjBggV69tlntWLFCj3xxBOH3QamqgIAAADAEaCtrU3btm1TU1OTJKmpqUnbtm1Te3v7YddNcAQAAACAI0Bra6umT58ux3EkSY7jaNq0aWptbT3sugmOAAAAAIBEBEcAAAAAOALU1dVpz549cl1XkuS6rvbu3au6urrDrpvgCAAAAABHgClTpqi+vl7Nzc2SpObmZtXX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\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()\n",
    "plt.xlim(40,80)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "fatal-classic",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:19.237384Z",
     "iopub.status.busy": "2021-04-30T11:28:19.236474Z",
     "iopub.status.idle": "2021-04-30T11:28:19.812508Z",
     "shell.execute_reply": "2021-04-30T11:28:19.811705Z"
    },
    "papermill": {
     "duration": 0.658644,
     "end_time": "2021-04-30T11:28:19.812684",
     "exception": false,
     "start_time": "2021-04-30T11:28:19.154040",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(80.0, 100.0)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()\n",
    "plt.xlim(80,100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "cooked-springfield",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:20.009831Z",
     "iopub.status.busy": "2021-04-30T11:28:19.989519Z",
     "iopub.status.idle": "2021-04-30T11:28:20.545006Z",
     "shell.execute_reply": "2021-04-30T11:28:20.544346Z"
    },
    "papermill": {
     "duration": 0.658925,
     "end_time": "2021-04-30T11:28:20.545175",
     "exception": false,
     "start_time": "2021-04-30T11:28:19.886250",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100.0, 512.3292)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 925.55x216 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "facet=sns.FacetGrid(train,hue='Survived',aspect=4)\n",
    "facet.map(sns.kdeplot,'Fare',shade=True)\n",
    "facet.set(xlim=(0,train['Fare'].max()))\n",
    "facet.add_legend()\n",
    "plt.xlim(100,train['Fare'].max())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "stunning-starter",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:20.707002Z",
     "iopub.status.busy": "2021-04-30T11:28:20.706157Z",
     "iopub.status.idle": "2021-04-30T11:28:20.720489Z",
     "shell.execute_reply": "2021-04-30T11:28:20.721050Z"
    },
    "papermill": {
     "duration": 0.097537,
     "end_time": "2021-04-30T11:28:20.721291",
     "exception": false,
     "start_time": "2021-04-30T11:28:20.623754",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "for dataset in train_test_data:\n",
    "  dataset.loc[dataset['Fare']<=15,'Fare']=1\n",
    "  dataset.loc[(dataset['Fare']>15) & (dataset['Fare']<=35),'Fare']=2\n",
    "  dataset.loc[(dataset['Fare']>35) & (dataset['Fare']<=100),'Fare']=3\n",
    "  dataset.loc[dataset['Fare']>100,'Fare']=4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "derived-camel",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:20.889194Z",
     "iopub.status.busy": "2021-04-30T11:28:20.888454Z",
     "iopub.status.idle": "2021-04-30T11:28:21.123598Z",
     "shell.execute_reply": "2021-04-30T11:28:21.124122Z"
    },
    "papermill": {
     "duration": 0.321752,
     "end_time": "2021-04-30T11:28:21.124341",
     "exception": false,
     "start_time": "2021-04-30T11:28:20.802589",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "bar_plot('Fare')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "cultural-crest",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:21.292248Z",
     "iopub.status.busy": "2021-04-30T11:28:21.291532Z",
     "iopub.status.idle": "2021-04-30T11:28:21.303847Z",
     "shell.execute_reply": "2021-04-30T11:28:21.304514Z"
    },
    "papermill": {
     "duration": 0.100443,
     "end_time": "2021-04-30T11:28:21.304759",
     "exception": false,
     "start_time": "2021-04-30T11:28:21.204316",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "C    59\n",
       "B    47\n",
       "D    33\n",
       "E    32\n",
       "A    15\n",
       "F    13\n",
       "G     4\n",
       "T     1\n",
       "Name: Cabin, dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "for dataset in train_test_data:\n",
    "  dataset['Cabin']=dataset['Cabin'].str[:1]\n",
    "train['Cabin'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "internal-establishment",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:21.471479Z",
     "iopub.status.busy": "2021-04-30T11:28:21.470734Z",
     "iopub.status.idle": "2021-04-30T11:28:21.752970Z",
     "shell.execute_reply": "2021-04-30T11:28:21.753515Z"
    },
    "papermill": {
     "duration": 0.367595,
     "end_time": "2021-04-30T11:28:21.753746",
     "exception": false,
     "start_time": "2021-04-30T11:28:21.386151",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEYCAYAAABMVQ1yAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Z1A+gAAAACXBIWXMAAAsTAAALEwEAmpwYAAAb/klEQVR4nO3de3RU9b338c9cMgOBhJAQYoBoAEWjLC+YI1bxsYXVlkOjeDtCs6TqAa3WCy7BkoMxsQShQ4JLWm+tIlSXl4IimOBDVBRQloslR/FpxCVeIpdDhJAQkkAyk7k8f0TSk5IwM8nOTGbP+/UX2Xvmt7/hm3zY/Oa397YEAoGAAAAxzxrtAgAAxiDQAcAkCHQAMAkCHQBMgkAHAJMg0AHAJOzRLuDo0ePy+825cjItbbDq6pqjXQZ6gN7FNjP3z2q1aOjQQV3ui3qg+/0B0wa6JFN/b2ZH72JbPPaPKRcAMImQztDdbreWLFmijz/+WE6nUxdffLFKSkpUXV2tgoICNTQ0KCUlRS6XS9nZ2X1cMgCgKyEFemlpqZxOpyorK2WxWHTkyBFJUnFxsfLz8zV9+nRt2LBBRUVFevHFF/u0YABA14IG+vHjx7V+/Xpt3bpVFotFkjRs2DDV1dVp9+7dWrVqlSQpLy9PJSUlqq+vV2pqat9WDQAh8Pm8Onq0Vl6vJ9qlhM1ud2jo0HTZbKF/1Bn0lfv371dKSoqefPJJ7dixQ4MGDdLcuXM1YMAAZWRkyGazSZJsNpuGDx+umpqasAI9LW1wyK+NRenpSdEuAT1E72JbenqSvvvuOw0aNEiDB4/oOCGNBYFAQE1Nx3TixFGNGTMm5PcFDXSfz6f9+/fr/PPP14IFC/T555/rrrvu0ooVK3pV8El1dc2m/TQ6PT1JtbVN0S4DPUDvYtvJ/h0/fkIZGcPk8wUkxVbODByYpEOHjp7yc2i1Wro9EQ66yiUzM1N2u115eXmSpIsuukhDhw7VgAEDdOjQIfl8PkntwX/48GFlZmb29vsAAMPE0pn5/9aTuoMGempqqiZOnKjt27dLkqqrq1VXV6fs7Gzl5OSooqJCklRRUaGcnBzmzwEgSiyhPOBi//79WrhwoRoaGmS32/XAAw/o6quv1rfffquCggI1NjYqOTlZLpcrrPkeKbJTLknJAzXA2fUsU6vbq6bGFkOPx3/bYxe9i20n+/fDD3t1xhlndWw/XQb0Rjj54fV6tXr183rvvXfkdDpktVo1YcK/6e6775Pd3rm2f61fOv2US0jfWVZWll566aVTto8dO1Zr164N6ZvoDwY47bpm3oYu95Uvny5+fQFzO10G9EY4+bFkyR/kdrfqhRdeUmLiIHm9Xm3c+JY8Hs8pgR6uqF/6DwDxYv/+fdq27QOtW/e2EhPb78dit9s1ffoNhozPpf8AECF79nylUaPOVHJycp+MT6ADgEkQ6AAQIePGnasDB/apsbGxT8Yn0AEgQrKyztSVV/4flZYu0YkTxyW1X8NTXr5eJ06c6PX4fCgKIG60ur0qXz69T8YNVWHhH/TCC3/Vf/7nLCUk2BUIBHT55VfK4XD0ug4CHUDcaGpsifry5ISEBP32t/fot7+9x/CxmXIBAJMg0AHAJAh0ADAJAh0ATIJABwCTINABwCRYtgggbgwd4pDd4TR8XK/HraPHov/cUgIdQNywO5z67rEbDR93zMNvSAoe6DfddI0cDoccDqc8HrcuuugSzZtX0Ovb5p5EoANABC1e7NKYMWfL5/Ppnnvu0Nat72vKlF8YMjZz6AAQBR6PRx6PW0lJxt1KlzN0AIigwsIFcjic+p//OaDLLpuoyy673LCxOUMHgAhavNil1atfUUXFu/J4PFqz5hXDxibQASAKnE6nrrjiKn3yyQ7DxiTQASAK/H6/du36b2VlnWnYmMyhA4gbXo/7xyWGxo8bqpNz6F5vm0aPHqvbbrvDsDoIdABxo/3in+hdAPT66+V9Oj5TLgBgEgQ6AJgEgQ4AJhHSHPrkyZPlcDjkdLbf1Gb+/Pm66qqrtGvXLhUVFcntdmvkyJEqLS1VWlpanxYMAOhayB+K/ulPf9K4ceM6vvb7/XrooYe0dOlS5ebm6umnn1ZZWZmWLl3aJ4UCAE6vx1MuVVVVcjqdys3NlSTNnDlTmzZtMqwwAEB4Qj5Dnz9/vgKBgC699FI9+OCDqqmp0YgRIzr2p6amyu/3q6GhQSkpKSEXkJY2OKyC+1J6elJMjInIoHexLT09SYcPW2W3//O8NXFwgpwJDsOP5W7z6ERzW9DXXXfdr+R0OpWQkKDW1haNHj1Ws2bdpgsvvKjL11ut1rB+DkMK9JdfflmZmZnyeDx67LHHtGjRIv385z8P+SCnU1fXLL8/YMhYwQT7i6mtbTL8eEaPicigd7HtZP/8fr+8Xn/HdmeCQzf//W7Dj7dmxjNq9IZ2cVFJyR81ZszZkqStW9/Xgw/ep+XLn9QFF4w/5bV+v/+Un0Or1dLtiXBIUy6ZmZmSJIfDofz8fH366afKzMzUwYMHO15TX18vq9Ua1tk5AMSzq6+erOnTb9Srr75kyHhBA/3EiRNqamr/FyIQCOjtt99WTk6Oxo8fr9bWVu3cuVOS9Nprr2nq1KmGFAUA8eL888fr+++/M2SsoFMudXV1uu++++Tz+eT3+zV27FgVFxfLarVq2bJlKi4u7rRsEQAQDuOmnIMGelZWltavX9/lvgkTJqi8vG/vTQAAZvbll7s1evRYQ8biSlEAiJIPP9yi9etf18yZtxgyHndbBBA3Wts8WjPjmT4ZN1SFhQuUkOBQa2uLsrNHq7R0RZcrXHqCQAcQN5oa3GpS6PcuNxq3zwUAhIRABwCTYMrlR36vp9srSb0e949POgGA/otA/5HV7tB3j93Y5b72ZxAS6AD6N6ZcAMAkCHQAMAmmXADEjaFJDtkHOA0f19vq1tGm6E/LEugA4oZ9gFPbp3f9WVlvXLnhDSmEQL/ppmvkcDjkcPzzH5WlS8uUmTniNO8KHYEOABG0eLGr437oRmMOHQBMgjN0AIigwsIFHVMuNptNK1ca83ALiUAHgIhiygUAEBSBDgAmwZQLgLjhbXW3LzHsg3FD9b/n0CWpoKBQ5513viF1EOgA4sbRJk9I68X7CvdDBwCEhEAHAJMg0AHAJAh0ADAJAh0ATIJABwCTYNkigLgxJHmgHE7jY8/j9upYY0tIr21sbNR11/27rr32ej3wwHxD6wjrO3vyySf15z//WeXl5Ro3bpx27dqloqIiud1ujRw5UqWlpUpLSzO0QAAwisNp16J5FYaPW7Q8L+TXvvvuJl1wwXi9916l7rlnrhISEgyrI+Qply+++EK7du3SyJEjJUl+v18PPfSQioqKVFlZqdzcXJWVlRlWGACY0caNb+nWW2dr7Nhz9OGHWw0dO6RA93g8WrRokR599NGObVVVVXI6ncrNzZUkzZw5U5s2bTK0OAAwk2+++VqNjcd06aX/pl/96hpt3PiWoeOHNOWyYsUKXXvttRo1alTHtpqaGo0Y8c/HJqWmpsrv96uhoUEpKSkhF5CWNjj0aqMoPT0pou9D9NG72JaenqTDh62y2yOz9iOU47z99luaNi1PCQk2TZ48RU88Uar6+iMaPnx4l6+3Wq1h/RwGDfTPPvtMVVVVmj/f2Mn7k+rqmuX3B/pk7H/Vm1/Q2tqmHh2vJ+9D9NG72Hayf36/X16vPyLHDHactrY2vfPO/1VCgkNvv13x4zavyss36NZbZ3f5Hr/ff8rPodVq6fZEOGigf/LJJ/r22281ZcoUSdIPP/yg2bNna9asWTp48GDH6+rr62W1WsM6OweAePHhh1uVlXWWnnlmZce2qqr/p8WLi7sN9HAFDfQ777xTd955Z8fXkydP1rPPPquzzz5ba9as0c6dO5Wbm6vXXntNU6dONaQoAOgLHrc3rBUp4YwbzMaNb+kXv/j3TtvGj79Qfr9fn33237rkkkt7XUePF2RarVYtW7ZMxcXFnZYtAkB/Fepa8b6wfPmfuty+Zs0Gw44RdqC///77HX+eMGGCysv79v6+AIDQcOk/AJgEgQ4AJkGgA4BJEOgAYBIEOgCYBLfPBRA3hiQ75HA6DR/X43brWKPH8HHDRaADiBsOp1NP/tftho9779JVkk4f6Hfccava2trk9bZp//59Gj16rCRp3LhztXBhsSF1EOgAEAHPPfc3SVJNzUHNmTNLq1e/YvgxmEMHAJMg0AHAJAh0ADAJAh0ATIJABwCTYJULgLjhcbt/XGJo/Lj9AYEOIG60X/wT3QuAMjNHaOPGzX0yNlMuAGASBDoAmASBDgAmQaADgEkQ6ABgEgQ6AJgEyxYBxI2hQwbK7jA+9rwer44eawn6uptuukYOh0MOR/s92SdMuFT33z/PsDoIdABxw+6w6+uyjwwf95z5k0J+7eLFLo0Zc7bhNUhMuQCAaXCGDgARVFi4oGPK5e6779PEiT8xbGwCHQAiqC+nXEIK9N/97nc6cOCArFarEhMT9cgjjygnJ0fV1dUqKChQQ0ODUlJS5HK5lJ2d3SeFAgBOL6RAd7lcSkpKkiS99957Wrhwod58800VFxcrPz9f06dP14YNG1RUVKQXX3yxTwsGAHQtpEA/GeaS1NzcLIvForq6Ou3evVurVrXfijIvL08lJSWqr69Xampq31QLAL3g9XjDWpESzrj9Qchz6A8//LC2b9+uQCCg559/XjU1NcrIyJDNZpMk2Ww2DR8+XDU1NQQ6gH4plLXifen118v7dPyQA/2xxx6TJK1fv17Lli3T3LlzDSkgLW2wIeP0tfT0pOAvMvB9iD56F9vS05N0+LBVdnvsrs62Wq1h/RyGvcrluuuuU1FRkc444wwdOnRIPp9PNptNPp9Phw8fVmZmZljj1dU1y+8PhFtGj/TmF7S2tqlHx+vJ+xB99C62neyf3++X1+uPdjk95vf7T/k5tFot3Z4IB/2n6/jx46qpqen4+v3339eQIUOUlpamnJwcVVRUSJIqKiqUk5PDdAsAREnQM/SWlhbNnTtXLS0tslqtGjJkiJ599llZLBY9+uijKigo0NNPP63k5GS5XK5I1BxxHl9bt2f3rW0eNTX0j+cJAohvQQN92LBhWrNmTZf7xo4dq7Vr1xpeVH/jsCXo5r/f3eW+NTOeUZMIdADRF7ufFgAAOuHSfwBxY8iQAXI4Egwf1+Np07FjrYaPGy4CHUDccDgStHz5csPHnTdvnqTQAt3r9epvf1up996rlM1ml81mU1ZWlmbPvkujR4/pVR0EOgBE0JIlf1Bra6v++te/KSkpSYFAQB9/vF379u0l0AEgVuzfv0/btn2gdeve7rilisVi0RVXGHM7Aj4UBYAI2bPnK40adaaSk5P7ZHwCHQCipLr6O912W75mzrxBTzxR1uvxCHQAiJBx487VgQP71NTUfjn/6NFjtHr1K/qP/5ih48ebez0+gQ4AEZKVdaYmTbpaLtdiNTf/M8BbWoy5CyQfigKIGx5P249LDI0fN1QPP/yoVq9+XnPm/EZ2u11JSUkaNixdt9xyW6/rINABxI32i3+iewFQQkKC7rjjbt1xR9e3E+kNplwAwCQ4Q+8lv8fT7Z0Y/R5PhKsBEM8I9F6yOhzaPv3GLvddueENiTsxAlEVCARksViiXUbYAoHwH/zDlAsA07LbHTp+vLFH4RhNgUBAx483ym53hPU+ztABmNbQoek6erRWzc0N0S4lbHa7Q0OHpof3nj6qBQCizmaza9iw8J5zHMuYcgEAkyDQAcAkCHQAMAkCHQBMgkAHAJMg0AHAJAh0ADAJAh0ATIJABwCTCHql6NGjR/X73/9e+/btk8Ph0FlnnaVFixYpNTVVu3btUlFRkdxut0aOHKnS0lKlpaVFom4AwL8IeoZusVg0Z84cVVZWqry8XFlZWSorK5Pf79dDDz2koqIiVVZWKjc3V2VlvX/IKQCgZ4IGekpKiiZOnNjx9cUXX6yDBw+qqqpKTqdTubm5kqSZM2dq06ZNfVcpAOC0wppD9/v9evXVVzV58mTV1NRoxIgRHftSU1Pl9/vV0NBgdI0AgBCEdbfFkpISJSYm6pZbbtG7775rSAFpaYMNGae/6u5pRuj/6F1si8f+hRzoLpdLe/fu1bPPPiur1arMzEwdPHiwY399fb2sVqtSUlLCKqCurll+f2RuPh+NBtfWNkX8mOi99PQkehfDzNw/q9XS7YlwSFMujz/+uKqqqvTUU0/J4Wh/gsb48ePV2tqqnTt3SpJee+01TZ061aCSAQDhCnqG/vXXX+svf/mLsrOzNXPmTEnSqFGj9NRTT2nZsmUqLi7utGwRABAdQQP9nHPO0VdffdXlvgkTJqi8vNzwogAA4eNKUQAwCQIdAEyCQAcAkyDQAcAkCHQAMAkCHQBMgkAHAJMg0AHAJAh0ADAJAh0ATIJABwCTINABwCTCesAFwuNt83V5D3aP26tjjS1RqAiAmRHofcieYNOieRWnbC9anheFagCYHVMuAGASBDoAmASBDgAmQaADgEkQ6ABgEgQ6AJgEgQ4AJkGgA4BJEOgAYBIEOgCYBIEOACZBoAOASRDoAGASQQPd5XJp8uTJOvfcc7Vnz56O7dXV1ZoxY4Z++ctfasaMGfr+++/7sk4AQBBBA33KlCl6+eWXNXLkyE7bi4uLlZ+fr8rKSuXn56uoqKjPigQABBc00HNzc5WZmdlpW11dnXbv3q28vPb7eufl5Wn37t2qr6/vmyoBAEH16AEXNTU1ysjIkM1mkyTZbDYNHz5cNTU1Sk1NDWustLTBPSkh5nX1JCP0L/QotsVj/6L+xKK6umb5/YGIHKs/Nbi2tinaJeA00tOT6FEMM3P/rFZLtyfCPVrlkpmZqUOHDsnn80mSfD6fDh8+fMrUDAAgcnoU6GlpacrJyVFFRfvzMisqKpSTkxP2dAsAwDhBp1wWL16sd955R0eOHNHtt9+ulJQUbdy4UY8++qgKCgr09NNPKzk5WS6XKxL1AgC6ETTQCwsLVVhYeMr2sWPHau3atX1SFAAgfFwpCgAmEfVVLoCZDB0yUHZH179WXo9XR4+1RLgixBMCHTCQ3WHX12UfdbnvnPmTIlwN4g1TLgBgEgQ6AJgEgQ4AJkGgA4BJEOgAYBIEOgCYBIEOACZBoAOASXBhEQD8KNav9CXQAeBHsX6lL1MuAGASBDoAmARTLjC1pBSnBiQ4utznc3tkc3a9z+/x9GVZiEFer7fb5xJ7PG06dqw1whWdikCHqQ1IcOjmv9/d5b41M57R9uk3drnvyg1vSHL3YWWINXa7XcuXL+9y37x58yRFP9CZcgEAk+AMHUBMGprkkH2As8t98TplRqADiEn2AU6mzP4FUy4AYBIEOgCYBIEOACZBoAOASRDoAGASrHKJAm9bW/dXnLndOtYYn0uuAPROrwO9urpaBQUFamhoUEpKilwul7Kzsw0ozbzsCQl68r9u73LfvUtXSSLQAYSv11MuxcXFys/PV2VlpfLz81VUVGREXQCAMPXqDL2urk67d+/WqlWrJEl5eXkqKSlRfX29UlNTQxrDarX0poSwDR86sNt99iHp3e5LT+z++3EO7/59Q7o5XlJKWrfvifTfidn1tHc97YM9ueurF3szJroW6f4lJycbfrxwne44lkAgEOjpwFVVVVqwYIE2btzYsW3atGkqLS3VBRdc0NNhAQA9wCoXADCJXgV6ZmamDh06JJ/PJ0ny+Xw6fPiwMjMzDSkOABC6XgV6WlqacnJyVFFRIUmqqKhQTk5OyPPnAADj9GoOXZK+/fZbFRQUqLGxUcnJyXK5XBozZoxR9QEAQtTrQAcA9A98KAoAJkGgA4BJEOgAYBIEOgCYBIEOACZBoAOASRDofeyNN96IdgkI4uDBg/r000/l8XS+bfH27dujVBHC8cEHH2jr1q2SpJ07d2rx4sVau3ZtlKuKDtah97Gf/vSn2rJlS7TLQDfeeustLVmyROnp6Wpubtbjjz+uSy65RJJ0/fXX680334xyhTidJ554Qtu3b5fX69Xll1+uqqoqXXXVVfrwww912WWX6b777ot2iRHFE4sMMHfu3C63BwIBHTt2LMLVIBwrV67Uhg0blJGRoR07dujBBx9USUmJJk2aJM51+r/Nmzdr/fr1amlp0aRJk7RlyxalpKTolltu0YwZMwh0hG/r1q1auHChEhISOm0PBALasWNHlKpCKAKBgDIyMiRJEydO1HPPPac777xTjzzyiCwW7l3e39ntdtlsNg0ePFhnnnmmUlJSJEmJiYmy2WzRLS4KCHQD5OTk6LzzztOFF154yr4VK1ZEoSKE4+R9iCTp7LPP1gsvvKA5c+bwv6sY4Pf7FQgEZLFYtGTJko7tgUBAXq83ipVFBx+KGqC4uLjbWwa/8sorEa4G4Zg1a5a++uqrTtuys7O1atUqXXHFFVGqCqGaP3++WltbJUnjx4/v2L53715df/310SoravhQFABMgjN0ADAJAh0ATIJABwCTINAN1NzcLL/fL0nas2ePNm7ceMrVh+if6F1so3/tCHQD/eY3v1Fra6tqa2s1e/ZsrVu3TkVFRdEuCyGgd7GN/rUj0A0UCASUmJioLVu26Oabb9bKlSv1xRdfRLsshIDexTb6145AN5Db7ZbH49H27dv1k5/8RJJktfJXHAvoXWyjf+3i7zvuQ9OmTdOVV16pAwcOaMKECaqtrZXT6Yx2WQgBvYtt9K8dFxYZ7NixY0pKSpLVatWJEyfU1NTUca8Q9G/0LrbRP87QDfXJJ5/IbrfLarVq7dq1+uMf/xiXn7THInoX2+hfOwLdQIsWLVJiYqK+/vprrVq1SiNGjNDDDz8c7bIQAnoX2+hfOwLdQHa7XRaLRdu2bdOvf/1r3XXXXWpsbIx2WQgBvYtt9K8dgW4gr9erzz//XO+++64uv/xySZLP54tyVQgFvYtt9K8dgW6guXPnqqioSBdddJHOOeccVVdX66yzzop2WQgBvYtt9K8dq1wAwCR4YpHBPvroI3355Zdyu90d2+69994oVoRQ0bvYRv8IdEOVlZXpH//4h7755htNmTJFmzdv7rhqDf0bvYtt9K8dc+gG2rp1q1auXKm0tDQtWrRI69at47mUMYLexTb6145AN5DD4ehYPtXW1qaMjAz98MMP0S4LIaB3sY3+tWPKxUCDBg1SS0uLLrnkEhUUFCg9PV0DBgyIdlkIAb2LbfSvHatcDHTkyBElJyfL5/Np1apVampq0qxZszRixIhol4Yg6F1so3/tCHQAMAmmXAxw//33y2KxdLt/xYoVEawG4aB3sY3+dUagG+BnP/tZtEtAD9G72Eb/OmPKxQA+n08ej0cDBw7stL2lpUUOh0M2my1KlSEYehfb6F9nLFs0QFlZmSoqKk7ZXlFRoeXLl0ehIoSK3sU2+tcZgW6AHTt26MYbbzxl+w033KBt27ZFoSKEit7FNvrXGYFuAJ/P1+UDaW0222k/sEH00bvYRv86I9AN0NraqpaWllO2Hz9+PC4fgxVL6F1so3+dEegGmDZtmhYsWKDm5uaObU1NTSosLNTUqVOjWBmCoXexjf51xioXA3i9XhUUFGjz5s3Kzs6WJH3//feaPHmyXC6X7HZWh/ZX9C620b/OCHQD7d27V7t375YknX/++XH5xJRYRe9iG/1rR6ADgEkwhw4AJkGgA4BJEOgAYBIEOgCYxP8HPsLVFRiwCJ8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "C1=train[train['Pclass']==1]['Cabin'].value_counts()\n",
    "C2=train[train['Pclass']==2]['Cabin'].value_counts()\n",
    "C3=train[train['Pclass']==3]['Cabin'].value_counts()\n",
    "df=pd.DataFrame([C1,C2,C3])\n",
    "df.index=['Class 1','Class 2','Class 3']\n",
    "df.plot.bar()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "democratic-things",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:21.919917Z",
     "iopub.status.busy": "2021-04-30T11:28:21.919168Z",
     "iopub.status.idle": "2021-04-30T11:28:21.933872Z",
     "shell.execute_reply": "2021-04-30T11:28:21.934513Z"
    },
    "papermill": {
     "duration": 0.099833,
     "end_time": "2021-04-30T11:28:21.934754",
     "exception": false,
     "start_time": "2021-04-30T11:28:21.834921",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cabin_mapping={\"A\":0,\"B\":1,\"C\":2,\"D\":3,\"E\":4,\"F\":5,\"G\":6,\"T\":7}\n",
    "for dataset in train_test_data:\n",
    "  dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)\n",
    "train['Cabin'].fillna(train.groupby(['Pclass'])['Cabin'].transform(\"median\"),inplace=True)\n",
    "test['Cabin'].fillna(test.groupby(['Pclass'])['Cabin'].transform(\"median\"),inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "subjective-sweet",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:22.113542Z",
     "iopub.status.busy": "2021-04-30T11:28:22.112790Z",
     "iopub.status.idle": "2021-04-30T11:28:22.125493Z",
     "shell.execute_reply": "2021-04-30T11:28:22.126194Z"
    },
    "papermill": {
     "duration": 0.104362,
     "end_time": "2021-04-30T11:28:22.126407",
     "exception": false,
     "start_time": "2021-04-30T11:28:22.022045",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "Drop_col=['Ticket','Parch','SibSp']\n",
    "train=train.drop(Drop_col,axis=1)\n",
    "test=test.drop(Drop_col,axis=1)\n",
    "train.drop('PassengerId',axis=1,inplace=True)\n",
    "train_data=train.drop('Survived',axis=1)\n",
    "Result=train['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "micro-regard",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:22.293063Z",
     "iopub.status.busy": "2021-04-30T11:28:22.292309Z",
     "iopub.status.idle": "2021-04-30T11:28:22.297846Z",
     "shell.execute_reply": "2021-04-30T11:28:22.298442Z"
    },
    "papermill": {
     "duration": 0.09139,
     "end_time": "2021-04-30T11:28:22.298657",
     "exception": false,
     "start_time": "2021-04-30T11:28:22.207267",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((891, 7), (891,))"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.shape,Result.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "running-colorado",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:22.473600Z",
     "iopub.status.busy": "2021-04-30T11:28:22.472492Z",
     "iopub.status.idle": "2021-04-30T11:28:22.479804Z",
     "shell.execute_reply": "2021-04-30T11:28:22.479077Z"
    },
    "papermill": {
     "duration": 0.09798,
     "end_time": "2021-04-30T11:28:22.480028",
     "exception": false,
     "start_time": "2021-04-30T11:28:22.382048",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def score(cm):\n",
    "  x=cm[0][0]+cm[1][1]\n",
    "  y=cm[1][0]+cm[0][1]\n",
    "  return (x/(x+y))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "velvet-certification",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:22.656432Z",
     "iopub.status.busy": "2021-04-30T11:28:22.655311Z",
     "iopub.status.idle": "2021-04-30T11:28:23.298463Z",
     "shell.execute_reply": "2021-04-30T11:28:23.299083Z"
    },
    "papermill": {
     "duration": 0.734652,
     "end_time": "2021-04-30T11:28:23.299338",
     "exception": false,
     "start_time": "2021-04-30T11:28:22.564686",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[483,  66],\n",
       "       [101, 241]])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "classifier=LogisticRegression()\n",
    "classifier.fit(train_data,Result)\n",
    "Pred=classifier.predict(train_data)\n",
    "from sklearn.metrics import confusion_matrix\n",
    "cm_logi=confusion_matrix(Result,Pred)\n",
    "cm_logi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "color-sauce",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:23.480243Z",
     "iopub.status.busy": "2021-04-30T11:28:23.479156Z",
     "iopub.status.idle": "2021-04-30T11:28:23.487689Z",
     "shell.execute_reply": "2021-04-30T11:28:23.487063Z"
    },
    "papermill": {
     "duration": 0.102525,
     "end_time": "2021-04-30T11:28:23.487859",
     "exception": false,
     "start_time": "2021-04-30T11:28:23.385334",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "81.25701459034792"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_logi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "divine-brooks",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:23.668876Z",
     "iopub.status.busy": "2021-04-30T11:28:23.668021Z",
     "iopub.status.idle": "2021-04-30T11:28:23.817538Z",
     "shell.execute_reply": "2021-04-30T11:28:23.818128Z"
    },
    "papermill": {
     "duration": 0.245219,
     "end_time": "2021-04-30T11:28:23.818341",
     "exception": false,
     "start_time": "2021-04-30T11:28:23.573122",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "81.25701459034792"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "cf_KNN=KNeighborsClassifier(n_neighbors=5)\n",
    "cf_KNN.fit(train_data,Result)\n",
    "Pred_KNN=cf_KNN.predict(train_data)\n",
    "cm_KNN=confusion_matrix(Result,Pred_KNN)\n",
    "cm_KNN\n",
    "score(cm_logi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "another-technology",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:23.999203Z",
     "iopub.status.busy": "2021-04-30T11:28:23.998376Z",
     "iopub.status.idle": "2021-04-30T11:28:24.002731Z",
     "shell.execute_reply": "2021-04-30T11:28:24.003248Z"
    },
    "papermill": {
     "duration": 0.099159,
     "end_time": "2021-04-30T11:28:24.003467",
     "exception": false,
     "start_time": "2021-04-30T11:28:23.904308",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "85.52188552188552"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_KNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "outdoor-biology",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:24.187604Z",
     "iopub.status.busy": "2021-04-30T11:28:24.186848Z",
     "iopub.status.idle": "2021-04-30T11:28:24.254323Z",
     "shell.execute_reply": "2021-04-30T11:28:24.253697Z"
    },
    "papermill": {
     "duration": 0.1642,
     "end_time": "2021-04-30T11:28:24.254486",
     "exception": false,
     "start_time": "2021-04-30T11:28:24.090286",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[465,  84],\n",
       "       [ 90, 252]])"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "cf_SVC=SVC(kernel='rbf')\n",
    "cf_SVC.fit(train_data,Result)\n",
    "Pred_SVC=cf_SVC.predict(train_data)\n",
    "cm_SVC=confusion_matrix(Result,Pred_SVC)\n",
    "cm_SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "brief-adolescent",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:24.425873Z",
     "iopub.status.busy": "2021-04-30T11:28:24.425025Z",
     "iopub.status.idle": "2021-04-30T11:28:24.431353Z",
     "shell.execute_reply": "2021-04-30T11:28:24.431866Z"
    },
    "papermill": {
     "duration": 0.094268,
     "end_time": "2021-04-30T11:28:24.432083",
     "exception": false,
     "start_time": "2021-04-30T11:28:24.337815",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "80.47138047138047"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_SVC)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "victorian-quarter",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:24.604552Z",
     "iopub.status.busy": "2021-04-30T11:28:24.603143Z",
     "iopub.status.idle": "2021-04-30T11:28:24.625144Z",
     "shell.execute_reply": "2021-04-30T11:28:24.624335Z"
    },
    "papermill": {
     "duration": 0.108368,
     "end_time": "2021-04-30T11:28:24.625328",
     "exception": false,
     "start_time": "2021-04-30T11:28:24.516960",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[401, 148],\n",
       "       [ 69, 273]])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "cf_NB=GaussianNB()\n",
    "cf_NB.fit(train_data,Result)\n",
    "Pred_NB=cf_NB.predict(train_data)\n",
    "cm_NB=confusion_matrix(Result,Pred_NB)\n",
    "cm_NB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "dependent-instrumentation",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:24.806657Z",
     "iopub.status.busy": "2021-04-30T11:28:24.805678Z",
     "iopub.status.idle": "2021-04-30T11:28:24.810783Z",
     "shell.execute_reply": "2021-04-30T11:28:24.811408Z"
    },
    "papermill": {
     "duration": 0.100816,
     "end_time": "2021-04-30T11:28:24.811665",
     "exception": false,
     "start_time": "2021-04-30T11:28:24.710849",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "75.64534231200898"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_NB)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "ethical-stylus",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:25.000269Z",
     "iopub.status.busy": "2021-04-30T11:28:24.999244Z",
     "iopub.status.idle": "2021-04-30T11:28:25.062653Z",
     "shell.execute_reply": "2021-04-30T11:28:25.061989Z"
    },
    "papermill": {
     "duration": 0.160567,
     "end_time": "2021-04-30T11:28:25.062858",
     "exception": false,
     "start_time": "2021-04-30T11:28:24.902291",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[527,  22],\n",
       "       [ 81, 261]])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "cf_DT=DecisionTreeClassifier()\n",
    "cf_DT.fit(train_data,Result)\n",
    "Pred_DT=cf_DT.predict(train_data)\n",
    "cm_DT=confusion_matrix(Result,Pred_DT)\n",
    "cm_DT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "starting-laugh",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:25.250520Z",
     "iopub.status.busy": "2021-04-30T11:28:25.249583Z",
     "iopub.status.idle": "2021-04-30T11:28:25.254851Z",
     "shell.execute_reply": "2021-04-30T11:28:25.254179Z"
    },
    "papermill": {
     "duration": 0.099479,
     "end_time": "2021-04-30T11:28:25.255035",
     "exception": false,
     "start_time": "2021-04-30T11:28:25.155556",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "88.43995510662177"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_DT)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "prostate-thought",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:25.432989Z",
     "iopub.status.busy": "2021-04-30T11:28:25.432262Z",
     "iopub.status.idle": "2021-04-30T11:28:25.510645Z",
     "shell.execute_reply": "2021-04-30T11:28:25.511240Z"
    },
    "papermill": {
     "duration": 0.171178,
     "end_time": "2021-04-30T11:28:25.511480",
     "exception": false,
     "start_time": "2021-04-30T11:28:25.340302",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[516,  33],\n",
       "       [ 74, 268]])"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "cf_RF=RandomForestClassifier(n_estimators=10)\n",
    "cf_RF.fit(train_data,Result)\n",
    "Pred_RF=cf_RF.predict(train_data)\n",
    "cm_RF=confusion_matrix(Result,Pred_RF)\n",
    "cm_RF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "cognitive-county",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:25.692644Z",
     "iopub.status.busy": "2021-04-30T11:28:25.691901Z",
     "iopub.status.idle": "2021-04-30T11:28:25.698655Z",
     "shell.execute_reply": "2021-04-30T11:28:25.697984Z"
    },
    "papermill": {
     "duration": 0.097846,
     "end_time": "2021-04-30T11:28:25.698818",
     "exception": false,
     "start_time": "2021-04-30T11:28:25.600972",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "87.99102132435466"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score(cm_RF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "included-weapon",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:25.886485Z",
     "iopub.status.busy": "2021-04-30T11:28:25.885700Z",
     "iopub.status.idle": "2021-04-30T11:28:25.893724Z",
     "shell.execute_reply": "2021-04-30T11:28:25.894296Z"
    },
    "papermill": {
     "duration": 0.108142,
     "end_time": "2021-04-30T11:28:25.894536",
     "exception": false,
     "start_time": "2021-04-30T11:28:25.786394",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,\n",
       "       1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,\n",
       "       1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,\n",
       "       1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0,\n",
       "       0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,\n",
       "       0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,\n",
       "       0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,\n",
       "       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,\n",
       "       0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,\n",
       "       1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,\n",
       "       1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,\n",
       "       0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,\n",
       "       1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data=test.drop('PassengerId',axis=1)\n",
    "Prediction=cf_RF.predict(test_data)\n",
    "Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "worthy-greensboro",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:26.083759Z",
     "iopub.status.busy": "2021-04-30T11:28:26.083017Z",
     "iopub.status.idle": "2021-04-30T11:28:26.096964Z",
     "shell.execute_reply": "2021-04-30T11:28:26.097527Z"
    },
    "papermill": {
     "duration": 0.109637,
     "end_time": "2021-04-30T11:28:26.097773",
     "exception": false,
     "start_time": "2021-04-30T11:28:25.988136",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived\n",
       "0            892         0\n",
       "1            893         0\n",
       "2            894         0\n",
       "3            895         0\n",
       "4            896         0\n",
       "..           ...       ...\n",
       "413         1305         0\n",
       "414         1306         1\n",
       "415         1307         0\n",
       "416         1308         0\n",
       "417         1309         1\n",
       "\n",
       "[418 rows x 2 columns]"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "submission = pd.DataFrame({\"PassengerId\": test[\"PassengerId\"], \"Survived\": Prediction})\n",
    "submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "tight-winter",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-04-30T11:28:26.283760Z",
     "iopub.status.busy": "2021-04-30T11:28:26.283002Z",
     "iopub.status.idle": "2021-04-30T11:28:26.301247Z",
     "shell.execute_reply": "2021-04-30T11:28:26.300289Z"
    },
    "papermill": {
     "duration": 0.114877,
     "end_time": "2021-04-30T11:28:26.301447",
     "exception": false,
     "start_time": "2021-04-30T11:28:26.186570",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived\n",
       "0          892         0\n",
       "1          893         0\n",
       "2          894         0\n",
       "3          895         0\n",
       "4          896         0"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "submission.to_csv('submission.csv', index=False)\n",
    "submission = pd.read_csv('submission.csv')\n",
    "submission.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 33.296382,
   "end_time": "2021-04-30T11:28:28.449569",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2021-04-30T11:27:55.153187",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
