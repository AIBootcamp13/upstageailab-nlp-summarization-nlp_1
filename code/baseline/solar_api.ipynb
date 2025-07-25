{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6n-Ps2nkVsBb"
   },
   "source": [
    "# **💁🏻🗨️💁🏻‍♂️대화 요약 SOLAR API code**\n",
    "> **Dialogue Summarization** 경진대회에 오신 여러분 환영합니다! 🎉    \n",
    "> 본 자료에서는 Solar Chat API를 이용하여 대화 요약 대회를 풀어봅니다.     "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kNq_LylZa1ug"
   },
   "source": [
    "## ⚙️ 데이터 및 환경설정"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MjCiuI_V4glr"
   },
   "source": [
    "### 1) 필요한 라이브러리 설치"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "VYqDF_-r2ToB"
   },
   "source": [
    "- 필요한 라이브러리를 설치한 후 불러옵니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "zbZ7SU9P2TYN"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "import time\n",
    "from tqdm import tqdm\n",
    "from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.\n",
    "from openai import OpenAI # openai==1.2.0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2) Solar Chat API Client 생성하기\n",
    "- 앞으로 Solar Chat API를 사용하기 위해 Client를 생성합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "UPSTAGE_API_KEY = \"up_*****************************\" # upstage.ai에서 발급받은 API KEY를 입력해주세요.\n",
    "\n",
    "client = OpenAI(\n",
    "    api_key=UPSTAGE_API_KEY,\n",
    "    base_url=\"https://api.upstage.ai/v1/solar\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-Qq46k6_CNQn"
   },
   "source": [
    "### 3) Solar Chat API 사용해보기 (선택)\n",
    "- 예시 코드를 통해 Solar Chat API를 사용해보세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 197,
     "referenced_widgets": [
      "e920dbc173c045d1a32143349f1dff8e",
      "58c794fb7ce543a39fdf66d757f6eeab",
      "8a6464a355f7464c989033965d418a8a",
      "3645438ace1f4596a8dbc157b48c1521",
      "58001a60eacc44d5b38a68648adccde4",
      "6f5fde5b0ac840a18bd5cc380e564ff6",
      "45187decb58b4ad39ad532259c6277e5",
      "2307c6dcbe0141acb5e61baae19cade7",
      "4747b668e2fa4ab58a449446f80030f5",
      "14f6c91d6c634379b498586c51e606e0",
      "08d05bc20a96432badd459e1ffaf868e",
      "5dfcf310ca9e4e2794076098a5d69cea",
      "3c284a826f6843f6aa47eacad478ac30",
      "6caedd60c6b747469c82930be1f95d6d",
      "64f2218f899d446393cfea44f206f0a6",
      "d068f541df3f438dbd5138863e64b2f2",
      "affff1d8a89e4c14955d1b2aa39ff1ab",
      "13651c09564a4337b8274c1cb436faa5",
      "3bcd6b6b956347b29e1efa20a1d00542",
      "2fd3d7bbcd6948d8904d33001f95ea03",
      "d22fbc2c5dbf422399e496c9b500025a",
      "775d8bbeceac4e2da4f21ab6235c89ed",
      "de1a3f7701c243839fe03b930a9b9e30",
      "ebc22683058a4f229c5588e52fc93536",
      "52095cc7087243ac916055e569fd22f3",
      "a15af9e8158f4903b9189f3d322a5ef3",
      "21d2e54b5a0a4f79973a512105da43eb",
      "083ea69907bb48d4a8fff919bac51aad",
      "2a190bda0b72407e9a953cd2104dd3b2",
      "c18f0e3bc35e44d9915c3f84cd282a26",
      "3a04e871b74b45d7bf02fd33bb103577",
      "ac00d6c2cf974b33a628acb3f1471316",
      "285007b45236478ca147c6df752c8da4"
     ]
    },
    "id": "gZOE9TInCQHJ",
    "outputId": "8ce58487-6199-408c-cb37-49af1e218bc2"
   },
   "outputs": [],
   "source": [
    "stream = client.chat.completions.create(\n",
    "    model=\"solar-1-mini-chat\",\n",
    "    messages=[\n",
    "      {\n",
    "        \"role\": \"system\",\n",
    "        \"content\": \"You are a helpful assistant.\"\n",
    "      },\n",
    "      {\n",
    "        \"role\": \"user\",\n",
    "        \"content\": \"Hello!\"\n",
    "      }\n",
    "    ],\n",
    "    stream=True,\n",
    ")\n",
    " \n",
    "for chunk in stream:\n",
    "    if chunk.choices[0].delta.content is not None:\n",
    "        print(chunk.choices[0].delta.content, end=\"\")\n",
    " \n",
    "# Use with stream=False\n",
    "# print(stream.choices[0].message.content)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "S2zt0b-8ogCL"
   },
   "source": [
    "### 4) 데이터 불러오기\n",
    "- 실험에서 쓰일 데이터를 load합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 293
    },
    "id": "QFHIE2G04y-K",
    "outputId": "19312d21-f5bf-495f-c626-cc17b82024a4"
   },
   "outputs": [],
   "source": [
    "# 데이터 경로를 지정해줍니다.\n",
    "DATA_PATH = \"../data/\"\n",
    "RESULT_PATH = \"./prediction/\"\n",
    "\n",
    "# train data의 구조와 내용을 확인합니다.\n",
    "train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))\n",
    "train_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 258
    },
    "id": "FAGaYvNZ09Sq",
    "outputId": "bf8bf286-19e7-469d-ffae-41e6ad795ae6"
   },
   "outputs": [],
   "source": [
    "# validation data의 구조와 내용을 확인합니다.\n",
    "val_df = pd.read_csv(os.path.join(DATA_PATH,'dev.csv'))\n",
    "val_df.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_IIaIrpH4kWo"
   },
   "source": [
    "## 1. Solar Chat API 요약 성능 확인하기\n",
    "- Solar Chat API을 이용하여 train 및 validation dataset에 포함된 dialogue 샘플을 요약해 봅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.\n",
    "rouge = Rouge()\n",
    "def compute_metrics(pred, gold):\n",
    "    results = rouge.get_scores(pred, gold, avg=True)\n",
    "    result = {key: value[\"f\"] for key, value in results.items()}\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "oWPawUUflwHa"
   },
   "outputs": [],
   "source": [
    "# Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.\n",
    "def build_prompt(dialogue):\n",
    "    system_prompt = \"You are an expert in the field of dialogue summarization. Please summarize the following dialogue.\"\n",
    "\n",
    "    user_prompt = f\"Dialogue:\\n{dialogue}\\n\\nSummary:\\n\"\n",
    "    \n",
    "    return [\n",
    "        {\n",
    "            \"role\": \"system\",\n",
    "            \"content\": system_prompt\n",
    "        },\n",
    "        {\n",
    "            \"role\": \"user\",\n",
    "            \"content\": user_prompt\n",
    "        }\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Solar Chat API를 활용해 Summarization을 수행하는 함수를 정의합니다.\n",
    "def summarization(dialogue):\n",
    "    summary = client.chat.completions.create(\n",
    "        model=\"solar-1-mini-chat\",\n",
    "        messages=build_prompt(dialogue),\n",
    "    )\n",
    "\n",
    "    return summary.choices[0].message.content"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (선택) parameter 변경하기\n",
    "- Solar Chat API를 사용할 때, parameter를 변경하여, 다양한 결과를 얻을 수 있습니다.\n",
    "- Parameter에 대한 자세한 설명은 [여기](https://developers.upstage.ai/docs/apis/chat#request-body)를 참고해주세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def summarization(dialogue):\n",
    "    summary = client.chat.completions.create(\n",
    "        model=\"solar-1-mini-chat\",\n",
    "        messages=build_prompt(dialogue),\n",
    "        temperature=0.2,\n",
    "        top_p=0.3,\n",
    "    )\n",
    "\n",
    "    return summary.choices[0].message.content"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Train Dataset을 이용하여 요약이 잘 되는지 확인해 봅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "6GDvodoF8sED"
   },
   "outputs": [],
   "source": [
    "# Train data 중 처음 3개의 대화를 요약합니다.\n",
    "def test_on_train_data(num_samples=3):\n",
    "    for idx, row in train_df[:num_samples].iterrows():\n",
    "        dialogue = row['dialogue']\n",
    "        summary = summarization(dialogue)\n",
    "        print(f\"Dialogue:\\n{dialogue}\\n\")\n",
    "        print(f\"Pred Summary: {summary}\\n\")\n",
    "        print(f\"Gold Summary: {row['summary']}\\n\")\n",
    "        print(\"==\"*50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    test_on_train_data()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Validation Dataset을 이용하여 요약을 진행하고, 성능을 평가해 봅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Validation data의 대화를 요약하고, 점수를 측정합니다.\n",
    "def validate(num_samples=-1):\n",
    "    val_samples = val_df[:num_samples] if num_samples > 0 else val_df\n",
    "    \n",
    "    scores = []\n",
    "    for idx, row in tqdm(val_samples.iterrows(), total=len(val_samples)):\n",
    "        dialogue = row['dialogue']\n",
    "        summary = summarization(dialogue)\n",
    "        results = compute_metrics(summary, row['summary'])\n",
    "        avg_score = sum(results.values()) / len(results)\n",
    "        \n",
    "        scores.append(avg_score)\n",
    "        \n",
    "    val_avg_score = sum(scores) / len(scores)\n",
    "\n",
    "    print(f\"Validation Average Score: {val_avg_score}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    validate(100) # 100개의 validation sample에 대한 요약을 수행합니다.\n",
    "    \n",
    "    # 전체 validation data에 대한 요약을 수행하고 싶은 경우 아래와 같이 실행합니다.\n",
    "    # validate() "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Solar Chat API로 요약하기\n",
    "- Solar Chat API을 이용하여 test dataset에 포함된 dialogue를 요약하고 제출용 파일을 생성합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def inference():\n",
    "    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))\n",
    "\n",
    "    summary = []\n",
    "    start_time = time.time()\n",
    "    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):\n",
    "        dialogue = row['dialogue']\n",
    "        summary.append(summarization(dialogue))\n",
    "        \n",
    "        # Rate limit 방지를 위해 1분 동안 최대 100개의 요청을 보내도록 합니다.\n",
    "        if (idx + 1) % 100 == 0:\n",
    "            end_time = time.time()\n",
    "            elapsed_time = end_time - start_time\n",
    "            \n",
    "            if elapsed_time < 60:\n",
    "                wait_time = 60 - elapsed_time + 5\n",
    "                print(f\"Elapsed time: {elapsed_time:.2f} sec\")\n",
    "                print(f\"Waiting for {wait_time} sec\")\n",
    "                time.sleep(wait_time)\n",
    "            \n",
    "            start_time = time.time()\n",
    "    \n",
    "    output = pd.DataFrame(\n",
    "        {\n",
    "            \"fname\": test_df['fname'],\n",
    "            \"summary\" : summary,\n",
    "        }\n",
    "    )\n",
    "    \n",
    "    if not os.path.exists(RESULT_PATH):\n",
    "        os.makedirs(RESULT_PATH)\n",
    "    output.to_csv(os.path.join(RESULT_PATH, \"output_solar.csv\"), index=False)\n",
    "\n",
    "    return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "-pJ1ZXf-5V50"
   },
   "outputs": [],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    output = inference()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "OsPmLfhbzZqS"
   },
   "outputs": [],
   "source": [
    "output  # 각 대화문에 대한 요약문이 출력됨을 확인할 수 있습니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Prompt Engineering\n",
    "- Prompt engineering을 통해 요약 성능 향상을 시도합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Few-shot prompt를 생성하기 위해, train data의 일부를 사용합니다.\n",
    "few_shot_samples = train_df.sample(1)\n",
    "\n",
    "sample_dialogue1 = few_shot_samples.iloc[0]['dialogue']\n",
    "sample_summary1 = few_shot_samples.iloc[0]['summary']\n",
    "\n",
    "print(f\"Sample Dialogue1:\\n{sample_dialogue1}\\n\")\n",
    "print(f\"Sample Summary1: {sample_summary1}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prompt를 생성하는 함수를 수정합니다.\n",
    "def build_prompt(dialogue):\n",
    "    system_prompt = \"You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue.\"\n",
    "\n",
    "    user_prompt = (\n",
    "        \"Following the instructions below, summarize the given document.\\n\"\n",
    "        \"Instructions:\\n\"\n",
    "        \"1. Read the provided sample dialogue and corresponding summary.\\n\"\n",
    "        \"2. Read the dialogue carefully.\\n\"\n",
    "        \"3. Following the sample's style of summary, provide a concise summary of the given dialogue.\\n\\n\"\n",
    "        \"Sample Dialogue:\\n\"\n",
    "        f\"{sample_dialogue1}\\n\\n\"\n",
    "        \"Sample Summary:\\n\"\n",
    "        f\"{sample_summary1}\\n\\n\"\n",
    "        \"Dialogue:\\n\"\n",
    "        f\"{dialogue}\\n\\n\"\n",
    "        \"Summary:\\n\"\n",
    "    )\n",
    "    \n",
    "    return [\n",
    "        {\n",
    "            \"role\": \"system\",\n",
    "            \"content\": system_prompt\n",
    "        },\n",
    "        {\n",
    "            \"role\": \"user\",\n",
    "            \"content\": user_prompt\n",
    "        }\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 변경된 prompt를 사용하여, train data 중 처음 3개의 대화를 요약하고, 결과를 확인합니다.\n",
    "if __name__ == \"__main__\":\n",
    "    test_on_train_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 변경된 prompt를 사용하여, validation data의 대화를 요약하고, 점수를 측정합니다.\n",
    "if __name__ == \"__main__\":\n",
    "    validate(100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "다른 방식으로 Few-shot sample을 제공하여 Prompt를 구성해 봅니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Few-shot sample을 다른 방식으로 사용하여 prompt를 생성합니다.\n",
    "def build_prompt(dialogue):\n",
    "    system_prompt = \"You are a expert in the field of dialogue summarization, summarize the given dialogue in a concise manner. Follow the user's instruction carefully and provide a summary that is relevant to the dialogue.\"\n",
    "\n",
    "    few_shot_user_prompt_1 = (\n",
    "        \"Following the instructions below, summarize the given document.\\n\"\n",
    "        \"Instructions:\\n\"\n",
    "        \"1. Read the provided sample dialogue and corresponding summary.\\n\"\n",
    "        \"2. Read the dialogue carefully.\\n\"\n",
    "        \"3. Following the sample's style of summary, provide a concise summary of the given dialogue. Be sure that the summary is simple but captures the essence of the dialogue.\\n\\n\"\n",
    "        \"Dialogue:\\n\"\n",
    "        f\"{sample_dialogue1}\\n\\n\"\n",
    "        \"Summary:\\n\"\n",
    "    )\n",
    "    few_shot_assistant_prompt_1 = sample_summary1\n",
    "    \n",
    "    user_prompt = (\n",
    "        \"Dialogue:\\n\"\n",
    "        f\"{dialogue}\\n\\n\"\n",
    "        \"Summary:\\n\"\n",
    "    )\n",
    "    \n",
    "    return [\n",
    "        {\"role\": \"system\", \"content\": system_prompt},\n",
    "        {\"role\": \"user\", \"content\": few_shot_user_prompt_1},\n",
    "        {\"role\": \"assistant\", \"content\": few_shot_assistant_prompt_1},\n",
    "        {\"role\": \"user\", \"content\": user_prompt},\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 변경된 prompt를 사용하여, train data 중 처음 3개의 대화를 요약하고, 결과를 확인합니다.\n",
    "if __name__ == \"__main__\":\n",
    "    test_on_train_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 변경된 prompt를 사용하여, validation data의 대화를 요약하고, 점수를 측정합니다.\n",
    "if __name__ == \"__main__\":\n",
    "    validate(100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (선택) 변경된 Prompt로 test dataset에 대한 요약을 진행합니다.\n",
    "- 변경된 prompt를 통해 점수가 개선되었다면, test dataset에 대한 요약을 진행하고 제출합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 변경된 prompt를 사용하여, test data의 대화를 요약하고, 결과를 확인합니다.\n",
    "if __name__ == \"__main__\":\n",
    "    output = inference()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "output"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "machine_shape": "hm",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.10.14"
  },
  "vscode": {
   "interpreter": {
    "hash": "d4d1e4263499bec80672ea0156c357c1ee493ec2b1c70f0acce89fc37c4a6abe"
   }
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "083ea69907bb48d4a8fff919bac51aad": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "08d05bc20a96432badd459e1ffaf868e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "13651c09564a4337b8274c1cb436faa5": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "14f6c91d6c634379b498586c51e606e0": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "21d2e54b5a0a4f79973a512105da43eb": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "2307c6dcbe0141acb5e61baae19cade7": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "285007b45236478ca147c6df752c8da4": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "2a190bda0b72407e9a953cd2104dd3b2": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "2fd3d7bbcd6948d8904d33001f95ea03": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "3645438ace1f4596a8dbc157b48c1521": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_14f6c91d6c634379b498586c51e606e0",
      "placeholder": "​",
      "style": "IPY_MODEL_08d05bc20a96432badd459e1ffaf868e",
      "value": " 295/295 [00:00&lt;00:00, 21.3kB/s]"
     }
    },
    "3a04e871b74b45d7bf02fd33bb103577": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "3bcd6b6b956347b29e1efa20a1d00542": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "3c284a826f6843f6aa47eacad478ac30": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_affff1d8a89e4c14955d1b2aa39ff1ab",
      "placeholder": "​",
      "style": "IPY_MODEL_13651c09564a4337b8274c1cb436faa5",
      "value": "tokenizer.json: 100%"
     }
    },
    "45187decb58b4ad39ad532259c6277e5": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "4747b668e2fa4ab58a449446f80030f5": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "52095cc7087243ac916055e569fd22f3": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_c18f0e3bc35e44d9915c3f84cd282a26",
      "max": 109,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_3a04e871b74b45d7bf02fd33bb103577",
      "value": 109
     }
    },
    "58001a60eacc44d5b38a68648adccde4": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "58c794fb7ce543a39fdf66d757f6eeab": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_6f5fde5b0ac840a18bd5cc380e564ff6",
      "placeholder": "​",
      "style": "IPY_MODEL_45187decb58b4ad39ad532259c6277e5",
      "value": "tokenizer_config.json: 100%"
     }
    },
    "5dfcf310ca9e4e2794076098a5d69cea": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_3c284a826f6843f6aa47eacad478ac30",
       "IPY_MODEL_6caedd60c6b747469c82930be1f95d6d",
       "IPY_MODEL_64f2218f899d446393cfea44f206f0a6"
      ],
      "layout": "IPY_MODEL_d068f541df3f438dbd5138863e64b2f2"
     }
    },
    "64f2218f899d446393cfea44f206f0a6": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_d22fbc2c5dbf422399e496c9b500025a",
      "placeholder": "​",
      "style": "IPY_MODEL_775d8bbeceac4e2da4f21ab6235c89ed",
      "value": " 682k/682k [00:00&lt;00:00, 5.40MB/s]"
     }
    },
    "6caedd60c6b747469c82930be1f95d6d": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_3bcd6b6b956347b29e1efa20a1d00542",
      "max": 682133,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_2fd3d7bbcd6948d8904d33001f95ea03",
      "value": 682133
     }
    },
    "6f5fde5b0ac840a18bd5cc380e564ff6": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "775d8bbeceac4e2da4f21ab6235c89ed": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "8a6464a355f7464c989033965d418a8a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_2307c6dcbe0141acb5e61baae19cade7",
      "max": 295,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_4747b668e2fa4ab58a449446f80030f5",
      "value": 295
     }
    },
    "a15af9e8158f4903b9189f3d322a5ef3": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_ac00d6c2cf974b33a628acb3f1471316",
      "placeholder": "​",
      "style": "IPY_MODEL_285007b45236478ca147c6df752c8da4",
      "value": " 109/109 [00:00&lt;00:00, 9.44kB/s]"
     }
    },
    "ac00d6c2cf974b33a628acb3f1471316": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "affff1d8a89e4c14955d1b2aa39ff1ab": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "c18f0e3bc35e44d9915c3f84cd282a26": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "d068f541df3f438dbd5138863e64b2f2": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "d22fbc2c5dbf422399e496c9b500025a": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "de1a3f7701c243839fe03b930a9b9e30": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_ebc22683058a4f229c5588e52fc93536",
       "IPY_MODEL_52095cc7087243ac916055e569fd22f3",
       "IPY_MODEL_a15af9e8158f4903b9189f3d322a5ef3"
      ],
      "layout": "IPY_MODEL_21d2e54b5a0a4f79973a512105da43eb"
     }
    },
    "e920dbc173c045d1a32143349f1dff8e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_58c794fb7ce543a39fdf66d757f6eeab",
       "IPY_MODEL_8a6464a355f7464c989033965d418a8a",
       "IPY_MODEL_3645438ace1f4596a8dbc157b48c1521"
      ],
      "layout": "IPY_MODEL_58001a60eacc44d5b38a68648adccde4"
     }
    },
    "ebc22683058a4f229c5588e52fc93536": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_083ea69907bb48d4a8fff919bac51aad",
      "placeholder": "​",
      "style": "IPY_MODEL_2a190bda0b72407e9a953cd2104dd3b2",
      "value": "special_tokens_map.json: 100%"
     }
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
