//Copyright 2015 Pareto Software, LLC, released under an MIT license: http://opensource.org/licenses/MIT
$( document ).ready(function() {
		//Inputs that determine what fields to show
		var docs = $('#live_form input:radio[name=docs]');			
		
		//Wrappers for all fields
		var pmids = $('#live_form textarea[name="pmids_box"]').parent();
		var term = $('#live_form textarea[name="term_box"]').parent();
		var pdfs = $('#live_form input[name="pdfs_box"]').parent();
		var all=pmids.add(term).add(pdfs);
		
		docs.change(function(){
			var value=this.value;						
			all.addClass('hidden'); //hide everything and reveal as needed
			
			if (value == 'pmids'){
				pmids.removeClass('hidden');								
			}
			else if (value == 'term'){
				term.removeClass('hidden');
			}		
			else if (value == 'pdfs'){
				pdfs.removeClass('hidden');
			}
		});	
});
