#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, session, request, redirect, flash
from getpage import getPage

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # note: valeur proposée dans la doc Flask


@app.route('/', methods=['GET'])
def index():
    # look for index.html in the templates folder and generate its content
    return render_template('index.html', message="Bonjour, monde !")


@app.route('/new-game', methods=['POST'])
def init_game():
    # initiate score at 0
    session['score'] = 0
    # get title of starting page
    page_title = request.form['start']
    # save it in the 'article' field of session
    session['article'] = page_title
    # redirect to "/game"
    return redirect('/game')


@app.route('/game', methods=['GET'])
def run_game():
    # recover page
    page = session['article']    
    # get title and content (links)
    session['title'], session['content'] = getPage(page) 
    # if the initial page does not exist, game is lost: display message
    if (session['score'] == 0) and (session['title'] is None):
        flash("Perdu!", 'defeat_1')
        flash("La page demandée n'existe pas.", 'defeat_2')
        flash("Refaite donc une partie avec une meilleure proposition.", 'defeat_3')        
        return redirect('/')         
    # if the initial page is Philosophie (or a page redirecting to it): game is lost, dispay message
    if (session['score'] == 0) and (session['title'] == 'Philosophie'):
        flash("Perdu!", 'defeat_1')
        flash("On ne peut pas commencer directement à la page philosophie.", 'defeat_2')
        flash("Refaite donc une partie sans prendre de raccourci.", 'defeat_3')
        return redirect('/')        
    # if there are no links in 'content', the game is lost: display message
    if not session['content'] :
        flash("Perdu!", 'defeat_1')
        flash("La page choisie ne contient aucun lien.", 'defeat_2')
        flash("Vous aurez peut être plus de chance la prochaine fois.", 'defeat_3')
        return redirect('/')
    # look for game.html in the templates folder and generate its content
    return render_template('game.html') # note: no need to pass title and links as arguments since they are session elements


@app.route('/move', methods=['POST'])
def move_game():
    # increment score by 1
    session['score'] += 1
    # collect the new page where the player wants to move
    new_page = request.form['destination']
    # the new page must normally be part of session['content'] (the list of possible moves given previous page)
    # if not, the player is playing on multiple tabs or manually submitting POST requests for impossible moves
    if new_page not in session['content']:
        flash("Perdu!", 'defeat_1')
        flash("Vous trichez. Soit en jouant sur plusieurs onglets, soit en envoyant", 'defeat_2')
        flash("des requêtes POST impossibles. Ce qui est répréhensible dans tous les cas.", 'defeat_3')
        return redirect('/')        
    # if the page is philosophie, display victory and move to index
    if new_page == "Philosophie":
        flash("Gagné!", 'victory_1')
        flash('Vous avez atteint la page "philosophie" en ' + str(session['score']) + ' coups.', 'victory_2')
        flash("Pourquoi ne pas rejouer avec une autre page?", 'victory_3')
        return redirect('/')
    # else, save page in session object
    session['article'] = new_page
    # return to /game
    return redirect('/game')


# Si vous définissez de nouvelles routes, faites-le ici

if __name__ == '__main__':
    app.run(debug=True)

