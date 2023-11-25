const searchBar = document.getElementById('search-bar');
const searchButton = document.getElementById('search-button');
const autoComplete = document.getElementById('auto-complete');
const className = 'selected';

var cursor = -1;
autoComplete.style.display = "none";

searchBar.onkeyup = (e)=> {
	var userData = e.target.value.trim();
	var matchList = [];
	var allList = autoComplete.querySelectorAll("li");

	if(e.key === 'ArrowDown' || e.keyCode === 40) {
		arrowDownListener(allList, className);
	}
	else if(e.key === 'ArrowUp' || e.keyCode === 38) {
		arrowUpListener(allList, className);
	}
	else if(userData) {
		cursor = -1;

		matchList = suggestions.filter((data) => {
			return data.toLocaleLowerCase().startsWith(userData.toLocaleLowerCase());
		});

		matchList = matchList.map((data) => {
			return data = '<li>' + data + '</li>';
		});

		autoComplete.style.display = "block";

		showSuggestions(matchList);
		allList = autoComplete.querySelectorAll("li");

		for (let i = 0; i < allList.length; i++) {
            allList[i].setAttribute("onclick", "select(this)");
        }		
	}
	else {
		var emptyList = [];
		showSuggestions(emptyList);
		autoComplete.style.display = "none";
	}
}

function arrowDownListener(allList, className) {
	cursor += 1;
	if(cursor > 0) {
		allList[cursor - 1].classList.remove(className);

		if(cursor > allList.length - 1) {
			cursor = 0;
		}
	}

	allList[cursor].classList.add(className);
	changeTextContent(allList[cursor]);	
}

function arrowUpListener(allList, className) {
	cursor -= 1;
	if(cursor != -2) {
		allList[cursor + 1].classList.remove(className);			
		if(cursor < 0) {
			cursor = allList.length - 1;
		}
	}
	else {
		cursor = allList.length - 1;
	}		

	allList[cursor].classList.add(className);
	changeTextContent(allList[cursor]);
}

function changeTextContent(element) {
    let selectData = element.textContent;
    searchBar.value = selectData;
}

function select(element){
	changeTextContent(element);
	autoComplete.style.display = "none";
}

function showSuggestions(list){
    var listData;
    if(!list.length){
        var userValue = searchBar.value;
        listData = '<li>'+ userValue +'</li>';
    }
    else{
    	var count = 0;
    	var countries = [];
    	for(let i = 0; i < list.length; i++) {
    		countries.push(list[i]);
    		count += 1;
    		if(count === 5) {
    			break;
    		}	
    	}   

    	listData = countries.join('');     
    }
    autoComplete.innerHTML = listData;
}