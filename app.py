
from flask import Flask, request, render_template, redirect, url_for
from flask import Flask, render_template, redirect, url_for, request
from model import *
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/link', methods=["POST", "GET"])
def link():
    if request.method == "POST":
        link_inp = request.form['linker']
        # print(type(link_inp))
    
        link_inp = link_inp.replace('.com', 'comkey')
        link_inp = link_inp.replace('https://', 'https')
        link_inp = link_inp.replace('www.', 'www')
        link_inp = link_inp.replace('/', 'slash')
        # print(link_inp)
        main = link_inp

        return redirect(url_for("preview_linker", linkage=main, tag='link_url'))
    else:
        return render_template("link.html")


@app.route('/text', methods=["POST", "GET"])
def pure_text():
    if request.method == "POST":
        inp_raw = request.form['raw_text']
        inp_raw = format_raw_text(inp_raw)
        return redirect(url_for('preview_linker', linkage=inp_raw, tag='pure_text'))
    else:
        return render_template("pure_text.html")


@app.route(f"/output/<tag>/<linkage>")
def preview_linker(linkage, tag):
    preview = linkage
    if tag == 'link_url':
        preview = preview.replace('https', 'https://')
        preview = preview.replace('www', 'www.')
        preview = preview.replace('slash', '/')
        preview = preview.replace('comkey', '.com')
        authart, publ, timg, allimg, tit, summ = meta_extract(preview)

    elif tag == 'pure_text':
        preview = preview.replace('uxd', ' ')     
        summ = preview  
        empty_msg = 'None'
        authart = empty_msg
        publ = empty_msg
        timg = empty_msg
        allimg = empty_msg
        tit = empty_msg

    sent = sentiment(summ)

    inp = tokenize_sequence(summ, token_basis)
    inp = inp[:600]
    inp = inp[None, :]
    # print(inp.shape)

    # feedforward_template.eval()
    # recurrent_template.eval()

    output_linear = '0 ERROR'
    output_lstm = '1 ERROR' #check for error without passing error

    output_linear = F.sigmoid(prediction(inp, feedforward)).round()
    # output_lstm = F.sigmoid(prediction(inp.long(), recurrent))

    all_types = list(pd.read_csv(data_dict['politifact_clean'])['veracity'].unique())

    if output_linear == 0:
        output_linear = f"Little Bias: Prediction = {output_linear}"
    elif output_linear == 1:
        output_linear = f"Substantial Bias: Prediction = {output_linear}"

    # statement_type = ''
    # if output_lstm <= 0.25:
    #     statement_type = 'True'
    # elif 0.25 < output_lstm <= 0.5:
    #     statement_type = 'Mostly True'
    # elif 0.5 < output_lstm <= 0.75:
    #     statement_type = 'Mostly False'
    # elif 0.75 < output_lstm <= 1:
    #     statement_type = 'False'
    # elif output_lstm > 1:
    #     statement_type = 'Pants on Fire!'

    # output_lstm = f"Veracity -> {statement_type}: {output_lstm}"

    # if output_lstm == 0:
    #     output_lstm = f"Limited Veracity: Prediction = {output_lstm}"
    # elif output_lstm == 1:
    #     output_lstm = f"Expressive Veracity: Prediction = {output_lstm}"

    

    

    return render_template("preview.html", preview_link=preview,
                                            author_article=authart, 
                                            published_article=publ,
                                            top_image = timg,
                                            all_image = allimg,
                                            title_article=tit,
                                            summary_article=summ,
                                            sentiment=sent,
                                            bias_point=output_linear,
                                            skew_point=output_lstm)



if __name__ == "__main__":
    app.run()

