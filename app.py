from flask import Flask, request, jsonify
import os
import json
from werkzeug.utils import secure_filename
from generate_storyboard import GPT_VISION, GPT, Encoder, Decoder, fetch_parchment, enchant, split_videos, summon
from extract_persona import PersonaAgent
from process_pdf import transcode_paperwork
from flask_cors import CORS, cross_origin

app=Flask(__name__)
CORS(app, support_credentials=True)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/pdf2text', methods=['POST'])
@cross_origin(supports_credentials=True)
def pdf2text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return transcode_paperwork(file_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/generate_ideas', methods=['POST'])
@cross_origin(supports_credentials=True)
def generate_ideas():
    if 'json_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    past_works = ""
    json_files = request.files.getlist('json_file')
    for json_file in json_files:
        if json_file and allowed_file(json_file.filename):
            json_filename = secure_filename(json_file.filename)
            json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            json_file.save(json_file_path)

            with open(json_file_path, 'r') as jf:
                data = json.load(jf)
                past_works = json.dumps(data) 

    ideas = enchant(past_works)
    response = jsonify(ideas)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


    
@app.route('/generate_storyboard', methods=['POST'])
@cross_origin(supports_credentials=True)
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        system_prompt = fetch_parchment('prompt.yml', 'standard')
        gpt = Encoder(GPT, system_prompt)
        gptV = Decoder(GPT_VISION, system_prompt)

        if filename.endswith('.txt'):
            with open(file_path, 'r') as f:
                brief = f.read()
                gpt_result = gpt.generate(brief)
            response = jsonify({"result": gpt_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        elif filename.endswith('.pdf'):
            encoded_images = split_videos(file_path, app.config['UPLOAD_FOLDER'])
            gptV_result = gptV.generate(encoded_images)
            gptV_result = summon(gptV_result)
            response = jsonify({"result": gptV_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/generate_storyboard_with_persona', methods=['POST'])
@cross_origin(supports_credentials=True)
def generate_storyboard_with_persona():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    
    # Retrieve the JSON file
    json_file = request.files.get('json_file')
    data_content = {}
    persona = ""
    if json_file:
        json_filename = secure_filename(json_file.filename)
        json_file_path = os.path.join('uploads', json_filename)
        json_file.save(json_file_path)

        # Open and read the JSON file
        with open(json_file_path, 'r') as jf:
            data_content = json.load(jf)
        
        persona = json.dumps(data_content)
    print("\n_____________________________________________________________\n")
    print("\n"+persona)
    print("\n_____________________________________________________________\n")

    file = request.files.get('file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        system_prompt = fetch_parchment('prompt.yml', 'personal_script')
        system_prompt = system_prompt + persona
        gpt = Encoder(GPT, system_prompt)
        gptV = Decoder(GPT_VISION, system_prompt)

        if filename.endswith('.txt'):
            with open(file_path, 'r') as f:
                brief = f.read()
                gpt_result = gpt.generate(brief)
            response = jsonify({"result": gpt_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        elif filename.endswith('.pdf'):
            encoded_images = split_videos(file_path, app.config['UPLOAD_FOLDER'])
            gptV_result = gptV.generate(encoded_images)
            gptV_result = summon(gptV_result)
            response = jsonify({"result": gptV_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response


        

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/storyboard_pipe', methods=['POST'])
@cross_origin(supports_credentials=True)
def pipe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    persona_file = request.files.get('persona')
    requirement_file = request.files.get('file')
    ideas_json_str = request.form.get('idea')

    persona_content = None
    ideas_content = []
    persona = ""
    idea = ""
    # Handling the persona JSON file
    if persona_file and allowed_file(persona_file.filename):
        filename = secure_filename(persona_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        persona_file.save(file_path)
        
        with open(file_path, 'r') as file:
            persona_content = json.load(file)

        persona = persona_content.get('Persona', 'No persona description available')
        os.remove(file_path)  # Optionally delete the file after loading
    
    print("\n_____________________________________________________________\n")
    print("\n"+persona)
    print("\n_____________________________________________________________\n")

    if ideas_json_str:
        ideas_content = json.loads(ideas_json_str)
        idea = json.dumps(ideas_content)

    if requirement_file and allowed_file(requirement_file.filename):
        filename = secure_filename(requirement_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        requirement_file.save(file_path)

        system_prompt = fetch_parchment('prompt.yml', 'pipe')
        system_prompt = system_prompt + "The user's persona is given here:" + persona + "/n Your generated content should be inspired by this idea: " + idea
        gpt = Encoder(GPT, system_prompt)
        gptV = Decoder(GPT_VISION, system_prompt)

        if filename.endswith('.txt'):
            with open(file_path, 'r') as f:
                brief = f.read()
                gpt_result = gpt.generate(brief)
            response = jsonify({"result": gpt_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        elif filename.endswith('.pdf'):
            encoded_images = split_videos(file_path, app.config['UPLOAD_FOLDER'])
            gptV_result = gptV.generate(encoded_images)
            gptV_result = summon(gptV_result)
            response = jsonify({"result": gptV_result})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/extract_persona', methods=['POST'])
@cross_origin(supports_credentials=True)
def extract_persona():
    data = request.data.decode('utf-8')

    if not data:
        return jsonify({"error": "No URL provided"}), 400

    url = data

    try:
        scrapper = PersonaAgent(url)

        video_content = ""
        if len(scrapper.transcripts) > 0:
            for i, transcript in enumerate(scrapper.transcripts):
                video_content += f"Video{i} Transcripts: {transcript} \n"
        html_text = scrapper.text

        system_prompt = fetch_parchment("prompt.yml", "persona")
        llm = Encoder(GPT, system_prompt)

        user_prompt = f"""
        Contents extracted from the content creator's personal webpage is shown below:
        Video Transcripts: {video_content}
        HTML Text: {html_text}
        """

        result = llm.generate(user_prompt)
        response = jsonify({"result": result})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response



    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    print("Server started. Listening on http://localhost:5500")
    app.run(host='0.0.0.0', port=5500,threaded=True, debug=True)